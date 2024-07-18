"""
Class to summarize the OpSim output
"""
from __future__ import absolute_import
__all__ = ['SynOpSim', 'PointingTree', 'add_simlibCols']
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import healpy as hp
from .opsim_out import OpSimOutput
from .trig import (convertToSphericalCoordinates,
                   convertToCelestialCoordinates,
                   angSep)

class SynOpSim(object):
    """Class with a summary of OpSim data base output.
    """
    def __init__(self,
                 pointings,
                 opsimversion='fbsv2',
                 raCol='ditheredRA',
                 decCol='ditheredDec',
                 angleUnit='degrees',
                 indexCol='obsHistID',
                 usePointingTree=False,
                 subset='None'):

        self.pointings = pointings
        self.raCol = raCol
        self.decCol = decCol
        self.angleUnit = angleUnit
        self.indexCol = indexCol
        self.subset = subset

        self.usePointingTree = usePointingTree
        self._pointingTree = None

    @staticmethod
    def df_subset_columns(df, subset):
        """Return the dataframe `df` with only a subset of the columns as
        specified in the list `subset`

        Parameters
        ----------
        df : `pd.DataFrame`
            the input dataframe to be returned with subset of the columns
        subset: (list of strings| 'all')
            if 'all', df is returned. Otherwise, return
            df[subset], with the same index.

        ..note: subset=[] returns only the index
        """
        if subset == 'all':
            # While we could be uniform and do according to the line below
            # there seems to be no reason to just return this
            # subset = df.columns
            return df

        # This protects us if `subset` is formed by taking `df.columns` and
        # performing some operations, even though the docstrings are specific
        # about `list`
        if isinstance(subset, pd.core.indexes.base.Index):
            subset = list(subset.values)

        # slightly roundabout method to work even if `subset` includes
        # name of index variable
        name = df.index.name
        if name in subset:
            return df.reset_index()[subset].set_index(name)
        else:
            return df[subset]


    @classmethod
    def fromOpSimDB(bls,
                    dbname,
                    subset='combined',
                    opsimversion='lsstv3',
                    zeroDDFDithers=True,
                    user_propIDs=None,
                    dithercolumns=None,
                    add_dithers=False,
                    tableNames=('Summary', 'Proposal'),
                    usePointingTree=False,
                    **kwargs):
        """
        Class Method to instantiate this from an OpSim sqlite
        database output which largely uses `OpSimOutput.fromOpSimDB`

        Parameters
        ----------
        dbname : string
            absolute path to database file
        subset : string, optional, defaults to 'combined'
            one of {'_all', 'unique_all', 'wfd', 'ddf', 'combined'}
            determines a sequence of propIDs for selecting observations
            appropriate for the OpSim database in use
        propIDs : sequence of integers, defaults to None
            proposal ID values. If present, overrides the use of subset
        tableNames : tuple of strings, defaults to ('Summary', 'Proposal')
            names of tables read from the OpSim database
        zeroDDFDithers : bool, defaults to True
            if True, set dithers in DDF to 0, by setting ditheredRA,
            ditheredDec to fieldRA, fieldDec
        opsimversion: {'lsstv3'|'sstf'|'lsstv4'|'fbsv1'| 'fbsv2'}
            version of OpSim corresponding to the output format.
        dithercolumns: `pd.DataFrame`, defaults to None
            a pandas dataframe with the columns `ditheredRA`, `ditheredDec` and
            index `obsHistID`
        add_dithers : Bool, defaults to False
            if add_dithers is True, the `ditheredRA`, `ditheredDec`
            columns will be removed and additional dithered columns
            either by `dithercolumns` or `get_dithercolumns` will
            be used.
        """
        if kwargs:
            opsout = OpSimOutput.fromOpSimDB(dbname,
                                             subset=subset,
                                             opsimversion=opsimversion,
                                             zeroDDFDithers=zeroDDFDithers,
                                             user_propIDs=user_propIDs,
                                             dithercolumns=dithercolumns,
                                             add_dithers=add_dithers,
                                             tableNames=tableNames,
                                             **kwargs)
        else:
            opsout = OpSimOutput.fromOpSimDB(dbname,
                                             subset=subset,
                                             opsimversion=opsimversion,
                                             zeroDDFDithers=zeroDDFDithers,
                                             user_propIDs=user_propIDs,
                                             dithercolumns=dithercolumns,
                                             add_dithers=add_dithers,
                                             tableNames=tableNames)

        opsimvars = OpSimOutput.get_opsimVariablesForVersion(opsimversion)

        raCol = opsimvars['pointingRA']
        decCol = opsimvars['pointingDec']
        angleUnit = opsimvars['angleUnit']
        indexCol = opsimvars['obsHistID']

        return bls(opsout.summary, opsimversion=opsimversion, raCol=raCol,
                   decCol=decCol, angleUnit=angleUnit, indexCol=indexCol,
                   usePointingTree=usePointingTree, subset=subset)

    @property
    def pointingTree(self):
        """
        if self.usePointingTree is False, this is set to None. Otherewise
        contains a `PointingTree` Object. This contains a `BallTree` of the
        pointings, and a method to find all pointings enclosed in a given radii. 
        """
        if self.usePointingTree is True:
            if self._pointingTree is None:
                self._pointingTree = PointingTree(self.pointings,
                                                  raCol='_ra',
                                                  decCol='_dec',
                                                  indexCol=self.indexCol,
                                                  leafSize=50)
        return self._pointingTree

    def pointingsEnclosing(self, ra, dec, circRadius=0., pointingRadius=1.75,
                           usePointingTree=None, transform=None, subset='all'):
        """
        Helper method returning a generator of pointings overlapping with
        circles of radius `circRadius around a sequence of positions given in
        terms of `ra` and `dec`. The calculation uses a `Tree` to make the
        calculations more efficient if `usePointingTree` is True, or uses direct
        calculations if this variable is set to `False`. A user may choose to
        obtain a subset of the `pointing` by supplying a subset in the form a
        list via the parameter `subset`.
 
        Parameters
        ----------
        ra : `np.ndarray` or a float, unit of degrees
            a float or an array of floats representing the ra values
        dec : `np.ndarray` or a float, unit of degrees
            a float or an array of floats representing the dec values
        circRadius: float, unit of degrees, defaults to 0.
            a circle around each of the 
        pointingRadius : degrees, defaults to 1.75
            radius of the field of view
        usePointingTree: {None|True|False}, defaults to `None`
            if None, usePointingTree = self.usePointingTree
            else the variable takes the {True|False} values assigned
        transform: function, Not implemented
        subset: (list of strings| 'all')
            if 'all', df is returned. Otherwise, return
            df[subset], with the same index.

        Returns
        -------
        A generator with the pointings that overlap with ra, dec 

        .. note: 1. the values in the generator may be accessed by next(generator)
            2. subset=[] returns only the index
        """
        print('check using ptree', usePointingTree)
        if transform is not None:
            raise NotImplementedError('transforms are not implemented yet')
        if usePointingTree is None:
            usePointingTree = self.usePointingTree

        if usePointingTree:
            hidxs = self.pointingTree.pointingsEnclosing(ra, dec, circRadius,
                                                         pointingRadius)
            for hidx in hidxs:
                yield self.df_subset_columns(self.pointings.loc[hidx], subset)
        else:
            print(self.pointings['_ra'].max())
            x = self.pointings[['_ra', '_dec']].copy().apply(np.degrees)
            pvecs = hp.ang2vec(x._ra, x._dec, lonlat=True)
            vecs = hp.ang2vec(ra, dec, lonlat=True)
            prad = np.radians(pointingRadius + circRadius)
            for vec in vecs:
                x.loc[:, 'dist'] = np.arccos(np.dot(pvecs, vec))
                idx = x.query('dist < @prad').index
                yield self.df_subset_columns(self.pointings.loc[idx], subset)


    def observedVisitsinRegion(self, nside=256, nest=True, minVisits=1,
                               maxVisits=None, outFile=None, writeFile=False):
        """
        return a `pd.DataFrame` with the healpixelID='hid', ra and dec of the
        healpixels 'hpix_ra', 'hpix_dec' in radians, and count the number of
        visits for all healpixels that have visits satisfying
        minVisits <= count <= maxVisits . This will also write out a csv
        outFile clobbering past version with this dataframe. 
        """
        if self.usePointingTree is False:
            raise NotImplementedError('This method works only with `PointingTree`')

        if writeFile:
            assert outFile is not None

        ipix = np.arange(hp.nside2npix(nside))
        hpix_ra, hpix_dec = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)

        # Input array for Tree query
        X = np.zeros(shape=(len(hpix_ra), 2))
        X[:, 0] = np.radians(hpix_dec)
        X[:, 1]= np.radians(hpix_ra)

        # count visits for each healpixel in the sky
        counts = self.pointingTree.tree.query_radius(X, r=np.radians(1.75),
                                                     count_only=True)
        survey = pd.DataFrame(dict(hid=ipix, ra=hpix_ra, dec=hpix_dec,
                                   numVisits=counts)).set_index('hid')

        # maxVisits is None, imply don't apply maxVisits 
        if maxVisits is None:
            maxVisits =  counts.max() + 1

        survey = survey.query('numVisits >= @minVisits and numVisits <=@maxVisits')
        if writeFile:
            survey.to_csv(outFile)
        return survey



    def sampleRegion(self, numFields=50000, minVisits=1, nest=True, nside=256,
                     rng=np.random.RandomState(1), outfile=None,
                     usePointingTree=True, subset='wfd', mwebv=0.):
        """This method samples a number `numFields` fields provided they have
        a minimal number of visits `minVisits`

        Parameters
        ----------
        numFields : int, mandatory
            Number of fields to sample
        minVisits : int, number, defaults to 1
            minimal number of visits required to consider the tile.
        nest : Bool, defaults to True
            use the `nest` method rather than `ring`
        nside : int, defaults to 256
            `Healpix.NSIDE`
        rng :
        outfile :
        usePointingTree : Bool, defaults to True
            whether to use PointingTree or not
        subset : {'wfd'|'ddf'|'combined'}
            which subset to use.
        """
        if subset == 'ddf':
            X = self.pointings[['_ra', '_dec']].drop_duplicates().apply(np.degrees)
            ra = X._ra.values
            dec = X._dec.values

            # WARNING: hp.ang2pix calls theta ra, and phi dec if lonlat is True
            hids = hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
            fieldIDs = hids
            counts = np.ones_like(hids)
            field = Field()
        else:
            if self.usePointingTree is False:
                raise NotImplementedError('This method works only with `PointingTree`')
            ipix = np.arange(hp.nside2npix(nside))
            hpix_ra, hpix_dec = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)
            X = np.zeros(shape=(len(hpix_ra), 2))
            X[:, 0] = np.radians(hpix_dec)
            X[:, 1]= np.radians(hpix_ra)
            counts = self.pointingTree.tree.query_radius(X, r=np.radians(1.75),
                                                         count_only=True)

            mask = counts > minVisits
            hids = ipix[mask]
            fieldarea = hp.nside2pixarea(nside, degrees=True)
    
            print('number of fields with visits above {0} is {1}'.format(minVisits,
                                                                         len(hids)))
            print('The total area of such fields is {} sq deg'.format(len(hids) * fieldarea))
            # theta, phi = convertToSphericalCoordinates(ra=self.pointings._ra,
            #                                           dec=self.pointings._dec,
            #                                           unit='radians')
            field = Field()
            # ipix = hp.ang2pix(nside=nside, theta=theta, phi=phi, nest=nest)
            # hid, count = np.unique(ipix, return_counts=True)
            # mask = count > minVisits - 1
            # hids = hid[mask]
    
            fieldIDs = rng.choice(hids, size=numFields, replace=False)
            ra, dec = hp.pix2ang(nside, fieldIDs, nest=nest, lonlat=True)
        pts = self.pointingsEnclosing(ra, dec, circRadius=0.,
                                      pointingRadius=1.75,
                                      usePointingTree=usePointingTree)


        # write out the survey files to an output file
        # if an outfile is provided

        # Skipping this for now
        # if outfile is not None:
        #    hdf_fname = outfile + '.hdf'
        #    survey = pd.DataFrame(dict(hid=hids, count=counts[hids]))
        #    survey.to_hdf(hdf_fname, key='survey')
        #    surveySample = pd.DataFrame(dict(fieldIDs=fieldIDs, ra=ra, dec=dec))
        #    surveySample.to_hdf(hdf_fname, key='surveySample')
        for i, fieldID in enumerate(fieldIDs):
            print(i, fieldID)
            field.setfields(fieldID, ra[i], dec[i],
                            next(pts).sort_values(by='expMJD'), mwebv=mwebv)
            yield field 



class Field(object):
    def __init__(self, fieldID=None, ra=None, dec=None, opsimtable=None,
                 mwebv=0.0):
        self.fieldID = fieldID
        self.ra = ra
        self.dec = dec
        self.mwebv = mwebv
        self.opsimtable = opsimtable

    def setfields(self, fieldID, ra, dec, opsimtable, mwebv=None):
        if mwebv is None:
            mwebv = mwebv
        self.fieldID = fieldID
        self.ra = ra
        self.dec = dec
        self.opsimtable = opsimtable


class PointingTree(object):
    def __init__(self,
                 pointings,
                 raCol='_ra',
                 decCol='_dec',
                 indexCol='obsHistID',
                 leafSize=50):
        """
        Create a tree of pointings

        Parameters
        ----------
        pointings : `pd.dataFrame` 
            of pointings with unique index values as the index column
        raCol :  string
            column name for a column holding ra values in radians
        decCol :  string
            column name for a column holding dec values in radians

        .. note : raCol and decCol are assumed to hold ra and dec in units of
        radians
        """
        self.pointings = pointings

        if self.validatePointings(pointings, raCol, decCol):
            self.raCol = raCol
            self.decCol = decCol
        else:
            raise ValueError('pointings, and the provided values of raCol, decCol {0}, {1} are incompatible'.format(raCol, decCol))

        # tree queries
        # Keep mapping from integer indices to obsHistID
        pointings.loc[:, 'intindex'] = np.arange(len(pointings)).astype(int)
        self.indMapping = pointings['intindex'].reset_index().set_index('intindex')

        # Build Tree
        self.tree = BallTree(pointings[[decCol, raCol]].values,
                             leaf_size=leafSize,
                             metric='haversine')

    @staticmethod
    def validatePointings(pointings, raCol, decCol):
        """
        Validates `pointings` as having required properties according to list
        of tests below

        Parameters
        ----------
        raCol : column name that should be in `pointings`
        decCol : column name that should be in `pointings`

        List of Tests:
        1. raCol and decCol should be in the columns of `pointings` 
        """
        cols = pointings.columns
        if raCol in cols and decCol in cols:
            assert pointings[raCol].max() < 2.0 * np.pi
            assert pointings[decCol].min() > - 1.0 * np.pi / 2.0
            return True
        else:
            print('the column name provided to PointingTree for ra and decs')
            print(' do not exist')
            return False

    def pointingsEnclosing(self, ra, dec, circRadius, pointingRadius=1.75):
        """
        Parameters
        ----------
        ra : float or sequence, degrees
            ra of the coordinates
        dec : float or sequence, degrees
            dec of the coordinates
        circRadius : degrees, mandatory
            radius of circle around point
        pointingRadius : degrees, defaults to 1.75
            radius of the field of view
        """
        # Treat only arrays
        ra = np.ravel(ra)
        dec = np.ravel(dec)

        assert len(ra) == len(dec)

        # flip conventions to dec, ra
        obj_posns = np.zeros(shape=(len(ra), 2))
        obj_posns[:, 0] = np.radians(dec)
        obj_posns[:, 1] = np.radians(ra)

        total_radius = np.radians(circRadius + pointingRadius)
        inds, dist = self.tree.query_radius(obj_posns,
                                            r=total_radius,
                                            count_only=False,
                                            return_distance=True)

        xx = self.indMapping
        return list(self.indMapping.obsHistID.loc[ptval].values for ptval in inds)




def add_simlibCols(opsimtable, pixSize=0.2):
    """
    Parameters
    ----------
    opsimtable: `~pandas.DataFrame` object, mandatory
        containing an opsim Output of version X. The main requirements here
        are that the columns 'finSeeing', 'fiveSigmaDepth', and
        'filtSkyBrightness' are defined. If the opsim output has differently
        named variables or transformed variables, these should be changed to
        meet the criteria.
    pixSize: float, optional, defaults to LSST value of 0.2
        pixel Size in units of arc seconds.


    Returns
    -------
    DataFrame with additional columns of 'simLibPsf', 'simLibZPTAVG', and
    'simLibSkySig' 

    .. note :: This was written from a piece of f77 code by David
        Cinabro sent by email on May 26, 2015. 
    """
    if 'finSeeing' in opsimtable.columns:
        psfwidth = 'finSeeing'
    else:
        psfwidth = 'FWHMeff'

    opsim_seeing = opsimtable[psfwidth] # unit of arc sec sq
    # magsky is in units of mag/arcsec^2
    # opsim_maglim is in units of mag
    opsim_maglim = opsimtable['fiveSigmaDepth']
    opsim_magsky = opsimtable['filtSkyBrightness']

    # Calculate two variables that come up in consistent units:
    # term1  = 2.0 * opsim_maglim - opsim_magsky

    # Area of pixel in arcsec squared
    pixArea = pixSize * pixSize

    term1 = 2.0 * opsim_maglim - opsim_magsky # * pixArea
    # term2 = opsim_maglim - opsim_magsky
    term2 = - (opsim_maglim - opsim_magsky) # * pixArea


    # Calculate SIMLIB PSF VALUE
    opsimtable.loc[:, 'simLibPsf'] = opsim_seeing /2.35 /pixSize
   
    # 4 \pi (\sigma_PSF / 2.35 )^2
    area = (1.51 * opsim_seeing)**2.
    
    opsim_snr = 5.
    arg = area * opsim_snr * opsim_snr

    # Background dominated limit assuming counts with system transmission only
    # is approximately equal to counts with total transmission
    zpt_approx = term1 + 2.5 * np.log10(arg)
    # zpt_approx = 2.0 * opsim_maglim - opsim_magsky + 2.5 * np.log10(arg)
    # ARG again in David Cinabro's code

    val = -0.4 * term2
    # val = -0.4 * (opsim_magsky - opsim_maglim)

    tmp = 10.0 ** val
    # Additional term to account for photons from the source, again assuming
    # that counts with system transmission approximately equal counts with total
    # transmission.

    zpt_cor = 2.5 * np.log10(1.0 + 1.0 / (area * tmp))
    simlib_zptavg = zpt_approx + zpt_cor
    # ZERO PT CALCULATION 
    opsimtable.loc[:, 'simLibZPTAVG'] = simlib_zptavg
    
    # SKYSIG Calculation
    npix_asec = 1. / pixSize**2.
    opsimtable.loc[:, 'simLibSkySig'] = np.sqrt((1.0 / npix_asec) \
    *10.0 **(-0.4 * (opsim_magsky - simlib_zptavg)))
    return opsimtable

class SummaryOpsim(object):
    
    
    def __init__(self, summarydf, user=None, host=None, survey='LSST',
                 telescope='LSST', pixSize=0.2, calculateSNANASimlibs=False):
        '''
        Create a summary of the OpSim output. 

        Parameters
        ----------
        summarydf: mandatory, `~pandas.DataFrame` (alternative constructors too)
            DataFrame corresponding to the opsim output (with appropriate
            selections) to be summarized.
        user: string, optional, defaults to None
            user running the program, used in writing out SNANA simlibs only 
            if None, the login name of the user is used.
        host: string, optional, defaults to None
            name of host machine, used only in writing out SNANA simlibs
            default of None assigns the output of `hostname` to this variable.
        survey: string, optional, defaults to 'LSST'
            name of survey, required only for writing out SNANA simlibs
        telescope: string, optional, defaults to 'LSST'
            name of survey, required only for writing out SNANA simlibs
        pixSize: float, optional, defaults to 0.2
            size of the pixel in arcseconds, defaults to 0.2 as appropriate
            for LSST
        calculateSNANASimlibs: Optional, Boolean, defaults to False
            Computes quantities only necessary for SNANA Simlib Calculation
        '''
        import os 
        import subprocess

        self.df = summarydf.copy(deep=True)
        self.calcMJDay(self.df)
        if 'simLibSkySig' not in self.df.columns:
            if calculateSNANASimlibs:
                self.df  = add_simlibCols(self.df)

        # SNANA has y filter deonoted as Y. Can change in input files to SNANA
        # but more bothersome.
        def capitalizeY(x):
            if 'y' in x:
                return u'Y'
            else:
                return x

        self.df.loc[:, 'filter'] = list(map(capitalizeY, self.df['filter']))
        self.minMJD = self.df.expMJD.min()
        self.minNight = self.df.night.min()
        self._fieldsimlibs = self.df.groupby(by='fieldID')
        self.fieldIds = self._fieldsimlibs.groups.keys()

        
        # report a user name, either from a constructor parameter, or login name
        if user is None:
            user = os.getlogin()
        self.user = user

        # report a host on which the calculations are done. either from
        # constructor parameters or from the system hostname utility 
        if host is None:
            proc = subprocess.Popen('hostname', stdout=subprocess.PIPE)
            host, err = proc.communicate()
        self.host = host

        self.telescope = telescope
        self.pixelSize = pixSize
        self.survey = survey

    @classmethod
    def fromOpSimASCII(cls, opSimFlatFile, **kwargs):
        """
        instantiates the sumamry table from opSim ASCII outputs like
        the outputs for version 2.168


        Parameters
        ---------
        opSimFlatFile : string, mandatory
            absolute path to ASCII file

        """
        import pandas as pd


        summary = pd.read_csv(opSimFlatFile, delimiter=r"\s+")
        return cls(summary, **kwargs)


    @classmethod
    def fromOpSimDB(cls, opSimDBfilename, tablename=None, sql_query=None,
                    dialect='sqlite', **kwargs):
        '''
        used to instantiate the summary from the opsim database


        Parameters
        ----------
        opSimDBfilename : str, mandatory
            absolute path to opsim sqlite database
        tablename : string, optional, defaults to None
            name of table in case it is not in sql_query
        sql_query : strng, mandatory
            sql_query to get the required OpSim visits
        dialect : string, defaults to 'sqlite'
            Only dialect supported at present

        Returns
        -------

        '''

        from sqlalchemy import create_engine
        import pandas as pd

        if dialect == 'sqlite':
            opSimDB = 'sqlite:///' + opSimDBfilename
        else:
            raise ValueError('other dialects not supported yet\n')
        engine = create_engine(opSimDB)
        if sql_query is None:
            summary = pd.read_sql_table(tablename, engine, **kwargs)
        else:
            summary = pd.read_sql_query(sql_query, engine, **kwargs)

        return cls(summary, **kwargs)
        
    def coords(self):

        ra = list(map(lambda x: self.ra(x), self.fieldIds))
        dec = list(map(lambda x: self.dec(x), self.fieldIds))

        return ra, dec

    @staticmethod
    def calcMJDay(data):
        """
        Adds an 'MJDay' column calculted from the expMJD variable
        """

        data.loc[:, 'MJDay'] = np.floor(data.expMJD.values)
        data.loc[:, 'MJDay'] = data['MJDay'].astype(int)



    def cadence_Matrix(self, summarydf=None, fieldID=None,
                       sql_query='night < 366', mjd_center=None,
                       mjd_range=[-50., 50.], observedOnly=False,
                       Filters=[u'u', u'g', u'r', u'i', u'z', u'Y'],
                       nightMax=365, nightMin=0):
    
        timeMin = nightMin
        timeMax = nightMax
        timeIndex = 'night'
        if mjd_center is not None:
            timeIndex = 'MJDay'
            if 'mjd' not in sql_query.lower():
                timeMin = mjd_center + mjd_range[0]
                timeMax = mjd_center + mjd_range[1]
                sql_query = 'MJDay > ' + str(timeMin) 
                sql_query += ' and MJDay < ' + str(timeMax)

        ss = pd.Series(np.arange(timeMin, timeMax))

        # group on filter and timeIndex (night)
        grouping_keys = ['filter', timeIndex]

        if summarydf is not None:
            data = summarydf
        else:
            data = self.simlib(fieldID)
            # queriedOpSim = self.simlib(fieldID).query(sql_query)

        if timeIndex == 'MJDay':
            # Check that input data has key MJDay
            if 'MJDay' not in data.columns:
                self.calcMJDay(data)

        queriedOpSim = data.query(sql_query)
        if len(queriedOpSim) == 0 :
            Matrix = pd.DataFrame(index=ss, columns=Filters)
            Matrix.fillna(0., inplace=True)
            return Matrix

        
        grouped = queriedOpSim.groupby(grouping_keys)
 
        # tuples of keys
        filts, times = zip( *grouped.groups.keys())

        # number of Observations in each group

        # Apparently the following does not work
        # numObs = grouped.apply(len).values

        # To do this correctly
        ks = grouped.groups.keys()
        numObs = np.array( map(lambda x: len(grouped.groups[x]), ks))

        # Create a new dataFrame with nights, Filters, numObs as cols
        cadence_dict = dict()
        cadence_dict['Filters'] = list(filts)
        cadence_dict[timeIndex] = list(times)

        # If observedOnly: set values above 1 to 1 
        if observedOnly:
            numObs = np.where(np.array(list(numObs)), 1, 0)
        
        cadence_dict['numObs'] = list(numObs)

        # pivot dataFrame to occupation numbers
        X = pd.DataFrame(cadence_dict)
        Matrix = X.pivot(timeIndex, 'Filters', 'numObs')

        # First make sure all filters are represented
        for filt in Filters:
            if filt not in Matrix.columns:
                Matrix[filt] = np.nan

        # reorder filters to u,g,r,i,z,y
        M = Matrix[Filters]
        
        # Extend to all values in plot
        Matrix = M.reindex(ss, fill_value=np.nan)

        return Matrix


    def mjdvalfornight(self, night):
        val = night + np.floor(self.minMJD) - self.minNight
        return np.int(val)

    def nightformjd(self, mjd) :
        val = mjd - np.floor(self.minMJD) + self.minNight
        return np.int(val)

    def cadence_plot(self, summarydf=None, fieldID=None,
                     racol=None, deccol=None,
                     sql_query='night < 366', mjd_center=None,
                     mjd_range = [-50, 50],
                     Filters=[u'u', u'g', u'r', u'i', u'z', u'Y'],
                     nightMin=0, nightMax=365, deltaT=5., observedOnly=False,
                     title=True, title_text=None, colorbar=True,
                     colorbarMin=0., showmjd=True, grid=True):
        """
        produce a cadence plot that shows the filters and nights observed in
        some subset of the opsim output time span for a field.


        Parameters
        ----------
        fieldID: integer, mandatory
            ID corresponding to the LSST Field as recorded as fieldID in OpSIM
            output
        sql_query: string, optional, defaults to obtaining the first season
            a sql query to select observations within the opsim summary object 
            associated with the field 
        Filters: list of strings, optional, defaults to LSST ugrizY
            a list of strings corresponding to filter names.

        """


        Matrix = self.cadence_Matrix(fieldID=fieldID, summarydf=summarydf,
                                     sql_query=sql_query,
                                     mjd_center=mjd_center, mjd_range=mjd_range,
                                     Filters=Filters, nightMin=nightMin,
                                      nightMax=nightMax, observedOnly=observedOnly)

        if mjd_center is not None:
            timeMin = mjd_center + mjd_range[0]
            timeMax = mjd_center + mjd_range[1]
        else:
            timeMin = nightMin
            timeMax = nightMax

        nightMatrix = Matrix.copy(deep=True)
        nightMatrix[nightMatrix > 0.5] = 1
        nightArray = nightMatrix.sum(axis=1).dropna()
        numNights = len(nightArray) 
        numFiltNights = int(nightArray.sum())
        numVisits = int(Matrix.sum().sum())

        if observedOnly:
            axesImage = plt.matshow(Matrix.transpose(), aspect='auto',
                                    cmap=plt.cm.gray_r, vmin=colorbarMin,
                                    vmax=1., extent=(timeMin - 0.5,
                                                     timeMax + 0.5,
                                                     -0.5, 5.5))
        else:
            axesImage = plt.matshow(Matrix.transpose(), aspect='auto',
                                    cmap=plt.cm.gray_r, vmin=colorbarMin,
                                    extent=(timeMin - 0.5, timeMax + 0.5,
                                            -0.5, 5.5))


        # setup matplotlib figure and axis objects to manipulate
        ax = axesImage.axes 

        # yticks: annotate with filter names
        # Note that it is also possible to get this from the DataFrame
        # by Matrix.columns, but the columns are sorted according to the order
        # in Filters
        
        ax.set_yticklabels(['0'] +Filters[::-1], minor=False)

        # Positiion x ticks at the bottom rather than top
        ax.xaxis.tick_bottom()
        ax.xaxis.get_major_formatter().set_useOffset(False)
        if mjd_center is not None:
            ax.axvline(mjd_center, color='r', lw=2.0)

        # Add a grid 
        # if mjd_center is not None:
        #    nightMin = mjd_center + mjd_range[0]
        #    nightMax = mjd_center + mjd_range[1]
        if grid:
            minorxticks = ax.set_xticks(np.arange(timeMin, timeMax,
                                                  deltaT), minor=True)

        # Hard coding this
        minoryticks = ax.set_yticks(np.arange(-0.5,5.6,1), minor=True)
        ax.set_adjustable('box-forced')
        ax.grid(which='minor')


        # If nightMin is different from 0, translate xticks
        # xticks = ax.get_xticks()
        # xticklabels = ax.get_xticklabels()
        # tmp = [item.get_text() for item in xticklabels]
        # xticklabel = np.array(map(eval, tmp[1:-1])) + nightMin
        # ax.set_xticks(xticks[1:-1])
        # ax.set_xticklabels(xticklabel)

        # Set a title with information about the field
        
        if title:
            # Make case for the time when fieldID is not supplied
            if fieldID is None:
                fieldIDval = 0 
                raval = summarydf[racol].iloc[0]
                decval = summarydf[deccol].iloc[0]
            else:
                fieldIDval = fieldID
                raval = self.ra(fieldID)
                decval = self.dec(fieldID)

            # Format field Info from attributes
            t_txt = 'fieldID: {:0>2d} (ra: {:+3.2f} dec: {:+3.2f}), visits: {:4d}, nights: {:3d}, nights in bands: {:3d}'
            t_txt = t_txt.format(fieldIDval, np.degrees(raval),
                                 np.degrees(decval), numVisits, numNights,
                                 numFiltNights)

            # if title_text is supplied use that instead
            if title_text is not None:
                t_txt = title_text
            ax.set_title(t_txt)

        # Set a colorbar
        if colorbar:
            plt.colorbar(orientation='horizontal')

        # Get the figure object
        fig = ax.figure

        return fig, Matrix, numVisits, numNights, numFiltNights


    def showFields(self, ax=None, marker=None, **kwargs):

        # ra needs to be in radians but after resetting offset
        ra = np.degrees(self.coords()[0])
        dec = self.coords()[1]

        x = np.remainder(ra + 360., 360.)
        ind  = x > 180.
        x[ind] -=360.

        ra = np.radians(x)
        # print ra
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='mollweide')
        if marker is None:
            marker = 'o'
        ax.scatter(ra, dec, marker=marker, **kwargs)
        ax.grid(True)
        fig  = ax.figure
        return fig


    def simlib(self, fieldID):

        return self._fieldsimlibs.get_group(fieldID)
    def ra(self, fieldID):
        ravals = np.unique(self.simlib(fieldID).fieldRA.values)
        if len(ravals)==1:
            return ravals[0]
        else:
            raise ValueError('The fieldDec of this group seems to not be unique\n')
        
    def dec(self, fieldID):
        decvals = np.unique(self.simlib(fieldID).fieldDec.values)
        if len(decvals)==1:
            return decvals[0]
        else:
            raise ValueError('The fieldDec of this group seems to not be unique\n')
    def meta(self, fieldID):
        meta = {}
        meta['LIBID'] = fieldID
        meta['dec'] = self.dec(fieldID)
        return meta
    
    def fieldheader(self, fieldID):
        
        ra = np.degrees(self.ra(fieldID))
        dec = np.degrees(self.dec(fieldID))
        mwebv = 0.01
        pixSize = self.pixelSize 
        nobs = len(self.simlib(fieldID))
        s = '# --------------------------------------------' +'\n' 
        s += 'LIBID: {0:10d}'.format(fieldID) +'\n'
        tmp = 'RA: {0:+10.6f} DECL: {1:+10.6f}   NOBS: {2:10d} MWEBV: {3:5.2f}'
        tmp += ' PIXSIZE: {4:5.3f}'
        s += tmp.format(ra, dec, nobs, mwebv, pixSize) + '\n'
        # s += 'LIBID: {0:10d}'.format(fieldID) + '\n'
        s += '#                           CCD  CCD         PSF1 PSF2 PSF2/1' +'\n'
        s += '#     MJD      IDEXPT  FLT GAIN NOISE SKYSIG (pixels)  RATIO  ZPTAVG ZPTERR  MAG' + '\n'
        return s
    
    def fieldfooter(self, fieldID):
        
        s = 'END_LIBID: {0:10d}'.format(fieldID)
        s += '\n'
        return s
        
    def formatSimLibField(self, fieldID, sep=' '):
    
        opSimSummary = self.simlib(fieldID)
        y =''
        for row in opSimSummary.iterrows():
            data = row[1] # skip the index
            # print(data['filter'], type(data['filter']), list(data['filter']))
            #       MJD EXPID FILTER 
            lst = ['S:',
                   "{0:5.4f}".format(data.expMJD),
                   "{0:10d}".format(data.obsHistID),
                   data['filter'], 
                   "{0:5.2f}".format(1.),                  # CCD Gain
                   "{0:5.2f}".format(0.25),                # CCD Noise 
                   "{0:6.2f}".format(data.simLibSkySig),   # SKYSIG
                   "{0:4.2f}".format(data.simLibPsf),      # PSF1 
                   "{0:4.2f}".format(0.),                  # PSF2 
                   "{0:4.3f}".format(0.),                  # PSFRatio 
                   "{0:6.2f}".format(data.simLibZPTAVG),   # ZPTAVG
                   "{0:6.3f}".format(0.005),               # ZPTNoise 
                   "{0:+7.3f}".format(-99.)]               # MAG
            s = sep.join(lst)
            y += s + '\n'
        return y
    
    def writeSimLibField(self, fieldID):
        s = self.fieldheader(fieldID)
        s += self.formatSimLibField(fieldID, sep=' ')
        s += self.footer(fieldID)
        return s

    def simLibheader(self): #, user=None, host=None, survey='LSST', telescope='LSST'):
        # comment: I would like to generalize ugrizY to a sort but am not sure
        # of the logic for other filter names. so ducking for now
        s = 'SURVEY: {0:}    FILTERS: ugrizY\n'.format(self.survey)
        s += 'USER: {0:}     HOST: {1:}\n'.format(self.user, self.host) 
        s += 'BEGIN LIBGEN\n'
        return s
    
    def simLibFooter(self):
        """
        """
        s = 'END_OF_SIMLIB:    {0:10d} ENTRIES'.format(len(self.fieldIds))
        return s


    def writeSimlib(self, filename, comments='\n'):
        with open(filename, 'w') as fh:
            simlib_header = self.simLibheader()
            simlib_footer = self.simLibFooter()
            fh.write(simlib_header)
            fh.write(comments)
            # fh.write('BEGIN LIBGEN\n')
            # fh.write('\n')
            for fieldID in self.fieldIds:
                fh.write(self.fieldheader(fieldID))
                fh.write(self.formatSimLibField(fieldID))
                fh.write(self.fieldfooter(fieldID))
            fh.write(simlib_footer)


