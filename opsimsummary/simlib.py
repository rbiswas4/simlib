#!/usr/bin/env python

"""
Module with functionality to represent SNANA simlib data.
"""
from __future__ import division, print_function, unicode_literals
__all__ = ['SimlibMixin', 'Simlibs']
import os
import numpy as np
import subprocess
from io import StringIO, BytesIO
from collections import OrderedDict
import pandas as pd
from .summarize_opsim import SynOpSim


class SimlibField(object):
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

class SimlibMixin(object):
    """
    Mixin for `SummaryOpsim` to provide the following additional functionality
    geared towards creating simlibs for SNANA.
    - Calculate additional columns for simlib either on a complete
        `OpSimOutput.summary` dataframe, or for a dataframe for a particular
        `patch` of sky (LIBID in the SNANA language).
    - Calculate variables required for SNANA simlib outside the Opsim data

    The parent class must have the following attributes:
        - subset (must be a valid string)
        The following attributes cannot be set by the user and are
        set by `simlibVars`
        - user (default can be None)
        - host (default can be None)
        - telescope
        - survey 
        - pixelSize
    In order to be able to write out the simlibs to disk, it should also have
    a method to provide a sequence (may be a generator) of fields, and
    `opsimtables`. The fields are instances of a class which has the following
    information

    `fieldID`
    `ra`
    `dec`
    `mwebv` (which may be set to a default value). The responsibility of
    selection of such fields and sorting out the correct requirements is
    of the parent class.
    """
    # Bad patch
    pixelSize = 0.2

    @property
    def simlibVars(self):
        """
        Collection of Attributes provided by the static method
        `self.get_simlibVars` as an ordered dict with the following keys
        (`user`, `host`, `pixelSize`, `survey`, `telescope`). Calling this.
        also sets class variables for each of these keys.
        """
        simlibVars = self.get_simlibVars(user=self.user,
                                         host=self.host,
                                         pixelSize=self.pixelSize,
                                         telescope=self.telescope,
                                         survey=self.survey)
        self.user = simlibVars['user']
        self.host = simlibVars['host']
        self.pixelSize = simlibVars['pixelSize']
        self.telescope = simlibVars['telescope']
        self.survey = simlibVars['survey']

        return simlibVars


    @staticmethod
    def get_simlibVars(user=None, host=None, pixelSize=0.2, telescope='LSST',
                       survey='LSST'):
        """ Computes quantities only necessary for SNANA Simlib Calculation
        Parameters
        ----------
        user: string, optional, defaults to None
            user running the program, used in writing out SNANA simlibs only
            if None, the login name of the user is used.
        host: string, optional, defaults to None
            name of host machine, used only in writing out SNANA simlibs
            default of None assigns the output of `hostname` to this variable.
        pixelSize: float, optional, defaults to 0.2
            size of the pixel in arcseconds, defaults to 0.2 as appropriate
            for LSST
        survey: string, optional, defaults to 'LSST'
            name of survey, required only for writing out SNANA simlibs
        telescope: string, optional, defaults to 'LSST'
            name of survey, required only for writing out SNANA simlibs
        """
        # report a user name, either from a constructor parameter, or login name
        if user is None:
            user = os.getlogin()

        # report a host on which the calculations are done. either from
        # constructor parameters or from the system hostname utility 
        if host is None:
            host = os.getenv( 'HOSTNAME' )
        x = (('user', user),
             ('host', host),
             ('pixelSize', pixelSize),
             ('survey', survey),
             ('telescope', telescope))
        return OrderedDict(x)

    def _capitalizeY(self, x):
        """private helper method
        """
        # SNANA has y filter deonoted as Y. Can change in input files to SNANA
        # but more bothersome.
        if 'y' in x:
            return u'Y'
        else:
            return x

    def preprocess_lib(self, opsimtable):
        """
        preprocess the dataframe with data for a single SNANA simlib
        field (ie. data corresponding to a single libid) if necessary.

        Parameters
        ----------
        opsimtable : `pd.DataFrame` with required data from OpSim corresponding
            to a single field.
        """
        # reasonable guess that columns have not been added
        if 'simLibSkySig' not in opsimtable.columns:
            df  = self.add_simlibCols(opsimtable, pixelSize=self.pixelSize)
            df['filter'] = list(map(self._capitalizeY, opsimtable['filter']))
        else:
            df = opsimtable
        return df

    @staticmethod
    def add_simlibCols(opsimtable, pixelSize=0.2):
        """
        Parameters
        ----------
        opsimtable: `~pandas.DataFrame` object, mandatory
            containing an opsim Output of version X. The main requirements here
            are that the columns 'finSeeing', 'fiveSigmaDepth', and
            'filtSkyBrightness' are defined. If the opsim output has differently
            named variables or transformed variables, these should be changed to
            meet the criteria.
        pixelSize: float, units of arc sec, defaults to LSST value of 0.2
            pixel Size


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
        pixArea = pixelSize * pixelSize

        term1 = 2.0 * opsim_maglim - opsim_magsky # * pixArea
        # term2 = opsim_maglim - opsim_magsky
        term2 = - (opsim_maglim - opsim_magsky) # * pixArea


        # Calculate SIMLIB PSF VALUE
        opsimtable['simLibPsf'] = opsim_seeing /2.35 /pixelSize

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
        opsimtable['simLibZPTAVG'] = simlib_zptavg


        # SKYSIG Calculation
        npix_asec = 1. / pixelSize**2.
        opsimtable['simLibSkySig'] = np.sqrt((1.0 / npix_asec) \
        * 10.0 **(-0.4 * (opsim_magsky - simlib_zptavg)))
        return opsimtable

    def fieldheader(self, fieldID, ra, dec, opsimtable, mwebv=0.0,
                    fieldtype=None, nobs_min_ddf=1100):
        """
        Parameters
        ----------
        fieldID : int
            integer for the unique field ID
        ra : float, degrees
            ra of the field location
        dec : float, degrees
            dec of the field location
        opsimtable : `np.array` of `pd.DataFrame`
            sequence of OpSim observations in above format to find number of
            observations.
        mwebv : float, defaults to 0.0
            milky way E(B-v) value. This is usually recomputed in SNANA
            depending on flags, and hence can be left as 0.0
        fieldtype : string, defaults to None
            string used to construct `Field: fieldtype` line, if None this
            line is left out.
        """
        nobs = len(opsimtable)
        # String formatting
        s = '# --------------------------------------------' +'\n' 
        s += 'LIBID: {0:10d}'.format(fieldID) +'\n'
        if fieldtype is not None:
            s += 'Field: {}\n'.format(fieldtype)
        tmp = 'RA: {0:+10.6f} DEC: {1:+10.6f}   NOBS: {2:10d}   MWEBV: {3:5.2f}'
        tmp += '   PIXSIZE: {4:5.3f}'
        # RKNOP 2024-07-17 : added the following line; the default
        #  nobs_min_ddf cutoff (1100) was based on looking at a histogram
        #  of nobs from baseline 3.4
        tmp += f'   FIELD: {"DDF" if nobs>=nobs_min_ddf else "WFD"}'
        s += tmp.format(ra, dec, nobs, mwebv, self.pixelSize) + '\n'
        # s += 'LIBID: {0:10d}'.format(fieldID) + '\n'
        s += '#                           CCD  CCD         PSF1 PSF2 PSF2/1' +'\n'
        s += '#     MJD      ID*NEXPOSE  FLT GAIN NOISE SKYSIG (pixels)  RATIO  ZPTAVG ZPTERR  MAG' + '\n'
        return s

    @staticmethod
    def fieldfooter(fieldID):

        s = 'END_LIBID: {0:10d}'.format(fieldID)
        s += '\n'
        return s

    def formatSimLibField(self, fieldID, opsimtable, sep=' '):

        opsimtable = self.preprocess_lib(opsimtable)
        y = ''
        for row in opsimtable.iterrows():
            data = row[1] # skip the index
            lst = ['S:',
                   "{0:5.4f}".format(data.expMJD),
                   "{0:10d}*2".format(data.obsHistID),
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
    
    def simlibFieldasString(self, fh, fieldID, ra, dec, opsimtable,
                            mwebv=0.0, fieldtype=None):

        opsimtable = opsimtable.reset_index()
        #raise NotImplementedError("Has not been checked")
        # Write out the header for each field
        s = self.fieldheader(fieldID, ra, dec, opsimtable,
                             mwebv=mwebv, fieldtype=fieldtype)
        # Write out the actual field
        s += self.formatSimLibField(fieldID, opsimtable, sep=' ')
        # Write out the footer for each field
        s += self.fieldfooter(fieldID)
        return s

    def simLibheader(self, numLibId=None, saturation_flag=1024,
                     doc='\n', comments='\n'):
        """
        return a string that is the header of the simlib file

        Parameters
        ----------
        numLibId : int, defaults to None
            number of libids in simlib
        saturation_flag : int, defaults to 1024
            value desired as saturation flag
        doc: string, defaults to `\n'
            contents of the DOCUMENTATION Block
        comments: string, defaults to `\n`
            comments passed on to `simlib` output
        """
        sv = self.simlibVars
        user = sv['user']
        host = sv['host'].splitlines()[0]
        # The decode lines below do the correct thing in py3
        # However the isinstance line does not, needs fixing
        # if isinstance(host, unicode):
        #    host = host.decode('utf-8')
        telescope = sv['telescope']
        survey = sv['survey']
        # comment: I would like to generalize ugrizY to a sort but am not sure
        # of the logic for other filter names. so ducking for now
        s = doc
        s += '\n\n\n'
        s += 'SURVEY: {0:}    FILTERS: ugrizY\n'.format(survey)
        s += 'USER: {0:}     HOST: {1}\n'.format(user, host) 
        if numLibId is not None:
            s += 'NLIBID: {}\n'.format(numLibId)
        s += 'NPE_PIXEL_SATURATE:   100000\n'
        s += 'PHOTFLAG_SATURATE:    {0}\n'.format(saturation_flag)
        s += comments + '\n'
        s += 'BEGIN LIBGEN\n'
        return s
    
    def simLibFooter(self, numFields):
        """
        """
        s = 'END_OF_SIMLIB:    {0:10d} ENTRIES'.format(numFields)
        return s


    def writeSimlib(self, filename, fields, doc='\n', comments='\n',
                    fieldtype=None, mwebv=0., numLibId=None):
            
        num_fields = 0
        with open(filename, 'w') as fh:
            # Write out the header to the simlib file
            simlib_header = self.simLibheader(numLibId=numLibId, doc=doc, comments=comments)
            fh.write(simlib_header)

            # Now write the actual simlib data to file
            for field in fields:

                # obtain the set of field dependent parameters from `SynOpSim`
                fieldID = field.fieldID
                ra = field.ra
                dec = field.dec
                mwebv = field.mwebv
                opsimtable = field.opsimtable

                fh.write(self.simlibFieldasString(self, num_fields, ra, dec,
                                                  opsimtable, mwebv=mwebv,
                                                  fieldtype=fieldtype))

                # Write out the header for each field
                # fh.write(self.fieldheader(num_fields, ra, dec, opsimtable,
                #                           mwebv=mwebv))
                # fh.write(self.formatSimLibField(fieldID, opsimtable))

                # Write out the footer for each field
                # fh.write(self.fieldfooter(fieldID))
                num_fields += 1

            # Now write out the footer to the entire simlib file 
            simlib_footer = self.simLibFooter(num_fields)
            fh.write(simlib_footer)
            return num_fields

class Simlibs(SynOpSim, SimlibMixin):
    """A class to write out simlibs to disk
    """
    pixelSize = 0.2
    host = None
    user = None
    telescope = 'LSST'
    survey = 'LSST'

    def simlibs_for_fields(self, surveyPix, mwebv=0.):
        """Generator for simlib fields for a sequence of fields
        defined in a dataFrame called `surveyPix`. The dataFrame
        `surveyPix` must have the following columns `simlibId`,
	`ra`, `dec` and must be sorted in increasing order of 
	`simlibId`.

	Parameters
	----------
	surveyPix : `pd.dataFrame`
            with the following columns `simlibId`, `ra`, `dec`
	mwebv : `np.float` defaults to 0.
	   A default value for the MW extinction


        Returns
        -------
        a generator of fields for the simlib file
        """

        surveyPix = surveyPix.reset_index().query('simlibId > -1').set_index('simlibId')
        ra = surveyPix.ra.values
        dec = surveyPix.dec.values
        pts = self.pointingsEnclosing(ra, dec, circRadius=0.,
                                      pointingRadius=1.75,
                                      usePointingTree=True)
        field = SimlibField()
        for i, fieldID in enumerate(surveyPix.reset_index().simlibId.values):
            field.setfields(fieldID, ra[i], dec[i],
                            next(pts).sort_values(by='expMJD'), mwebv=mwebv)
            yield field

    def get_surveyPix(self, surveydf, numFields=15, rng=np.random.RandomState(0)):
        """ Get a random selection of survey pixels observed that have numbers
	of visits in between the min and max visits.

	Parameters
	----------
	surveydf : `pd.DataFrame`
	    a pandas dataframe with a collection of selected fields with at
            least the  the following columns a unique index `hid` for each
            field, an `ra`, and a `dec`
	numFields : integer, defaults to 15
	    number of samples of fields desired.
	rng : instance of `np.random.RandomState`, defualts to using 0 as seed
	   a random state.


	Returns
	-------
	a dataframe with at least `hid` the original index for each field, `ra`,
	`dec`, and `simlibID` sorted by `simlibId`. This dataframe contains a
	mapping from the new index `simlibId` to the old index `hid`
	"""
        surveydf['simlibId'] = -1

        if numFields <= len(surveydf):
            surveydf = surveydf.sample(n=numFields, replace=False,
                                       random_state=rng)
            # hids = rng.choice(surveydf.reset_index()['hid'].values, size=numFields,
            #               replace=False)
        else:
            surveydf = surveydf.sample(n=numFields, replace=True,
                                       random_state=rng)
            print("Warning: You have asked for more samples than the original number of fields")
            print('Printing original number of fields instead')

        hids = surveydf.reset_index()['hid'].values

        surveydf.reset_index().set_index('hid')
        surveydf.loc[hids, 'simlibId'] = np.arange(len(hids))
        return surveydf

    def randomSimlibs(self, numFields=50, fname='test.simlib',
                      rng=np.random.RandomState(1), outfile=None,
                      mapping_outfile='mapping.csv', mwebv=0.,
                      fieldtype=None, minVisits=1):

        if fieldtype is None:
            fieldtype = self.subset.upper()


        if outfile is None:
            outfile = fname  + '.hdf'
        fields = self.sampleRegion(numFields=numFields, rng=rng,
                                   outfile=outfile, subset=self.subset,
                                   minVisits=minVisits, nside=256,
                                   mwebv=mwebv)
        num_fields = self.writeSimlib(fname, fields, fieldtype=fieldtype, mwebv=mwebv)

        fields = self.sampleRegion(numFields=numFields, rng=rng,
                                   outfile=outfile, subset=self.subset)
        df = pd.DataFrame(dict(SNANAID=np.arange(num_fields),
                               healpixID=list(field.fieldID for field in fields
                                             )))
        df.to_csv(mapping_outfile)

class Simlib(object):

    def __init__(self, simlibDict, simlibMetaData=None):


        self.simlibDict = simlibDict
        self.fieldIDs = self.simlibDict.keys()
        if simlibMetaData is not None:
            self.meta = simlibMetaData

    @classmethod
    def fromSimlibFile(cls, simlibFileName):
        '''
        Constructor for class using an ASCII 

        Parameters
        ----------
        simlibFileName: string, mandatory
            absolute path to SNANA simlib file

        Returns
        -------
        dictionary with LIBIDs (fieldIDs) of type int as keys, and the
        corresponding FieldSimlib objects as values

        Examples
        --------
        >>> sl = Simlib.fromSimlibFile(simlibFileName)
        '''

        file_header, file_data, file_footer = cls.read_simlibFile(simlibFileName)
        mydict = cls.getSimlibs(file_data)
        meta = cls.simlibMetaData(file_header) 
        cls = cls(simlibDict=mydict, simlibMetaData=meta)
        cls.validate(file_footer)

        return cls

    def validate(self, file_footer):
        '''
        '''
        numberlist = list(filter(lambda x: x.isdigit(), file_footer.split()))
        if len(numberlist) !=1:
            raise ValueError('There should only be one integer in the footer')
        numLibId = int(numberlist[0])

        if numLibId != len(self.fieldIDs):
            raise ValueError('The number of fieldIDs is in the simlib does not match the number stated in footer')

        return

    @staticmethod
    def simlibMetaData(simlibHeader):
        '''
        parse the string corresponding to the header of a SNANA simlib file to
        get the simlib MetaData, stored in the form of a string valued
        dictionary with the following keys:
        'SURVEY', 'FILTERS', 'HOST', 'USER', 'COMMENT'

        Parameters
        ----------
        simlibHeader: string corresponding the header of an SNANA simlib file as
            parsed by cls.read_simlibFile.
        Returns
        -------
        dictionary of keys above and values.
        '''

        comments = []
        fields = []

        lines = simlibHeader.split('\n')
        for line in lines:
            if line.startswith('COMMENT') or line.startswith('#'):
                comments.append(line)
            else:
                fields.append(line)
        ss = ' '.join(fields)
        words = ss.split()
        keys = list(map(lambda x: x[:-1], words[0::2]))
        vals = words[1::2]
        if len(keys) != len(vals):
            raise ValueError('the numberof fields in dict should match vals')

        meta = dict(zip(keys, vals))
        meta['COMMENTS'] = '\n'.join(comments)

        return meta


    def simlibData(self, fieldID):
       return self.simlibDict[fieldID].data

    @classmethod
    def getSimlibs(cls, file_data):
    # def getSimlibs(cls, simlibFile):
        # file_header, file_data, file_footer = cls.read_simlibFile(simlibFile)
        simlibStrings = cls.split_simlibStrings(file_data)
        mydict = dict()
        for strings in simlibStrings:
            s = FieldSimlib.fromSimlibString(strings)
            mydict[s.fieldID] = s

        return mydict 
            

    @staticmethod
    def read_simlibFile(simlibfile):
    
        # slurp into a string
        with open(simlibfile) as f:
            ss = f.read()
        
        # split into header, footer and data
        fullfile = ss.split('BEGIN LIBGEN')
        file_header = fullfile[0]
        if 'END_OF_SIMLIB' in ss:
            data, footer = fullfile[1].split('END_OF_SIMLIB')
        else:
            data = fullfile[1]
            footer = ''

        return file_header, data, footer

    @staticmethod
    def split_simlibStrings(simlibStrings):
        simlibs = simlibStrings.split('\nLIBID')[1:]
        simlibs = map(lambda x: 'LIBID' + x.split('# -')[0], simlibs)
        return simlibs
class FieldSimlib(object):
    """
    Class to hold data corresponding to a particular fieldID (LIBID) of a
    SNANA SIMLIB file and methods. The fieldSimlib class for a particular field
    has the following attributes, and may be instantiated by supplying these, 
    or can be conveniently constructed from the string corresponding to the
    description of this data in an SNANA simlib file using the constructor
    fromSimlibString:


    Parameters
    ----------
    simlibdata : a pandas dataFrame
    simlib_meta : a dictionary


    Attributes
    ----------
    fieldID : int
        a unique integer identifying the field of view. Different pointings
        of the same field at different times (but possibly with dithers) are
        associated with the same fieldID
    meta : dict
        metadata associated with the field, which has at least the following
        keys:
        LIBID, RA, DEC, MWEBV, NOBS, PIXSIZE
    data : `~pd.DataFrame` object with the observations and having at least the
        following columns: 'MJD', 'IDEXPT', 'FLT', 'GAIN', 'NOISE', 'SKYSIG',
        'PSF1', 'PSF2', 'PSFRatio', 'ZPTAVG', 'ZPTERR', 'MAG']. The meanings of
        these columns are discussed in the SNANA manual in the sub-section
        'The 'SIMLIB' Observing file (4.7)
    """
    def __init__(self, simlibdata, simlib_meta):
        """
        Instantiate the class from the basic data
        """ 

        self.data = simlibdata 
        self.meta = simlib_meta
        self.fieldID = self.meta['LIBID']

    @classmethod
    def fromSimlibString(cls, simlibstring):
        '''
        Basic constructor method to take a string corresponding to a
        field simlib data corresponding to a single LIBID and parse it to
        metadata containing the properties of the field, a
        `~pandas.DataFrame` containing the data, and the string after the
        data

        Parameters
        ----------
        simlibstring : string, mandatory
        '''

        # split into three parts
        header, data, footer = cls.split_simlibString(simlibstring)

        # create the DataFrame
        clsdata = cls.simlibdata(data)

        # parse header to get header metadata and header fields
        header_metadata, header_fields = cls.split_header(header)
        clsmeta = cls.libid_metadata(header_metadata)

        # Instantiate the class and make sure it works
        myclass = cls(simlibdata=clsdata, simlib_meta=clsmeta) 
        myclass.validate(footer)
        return myclass 

    def validate(self, validate_string):
        """
        Validate the interpretation of the field simlib data from a field simlib
        string by checking 1. the LIBID at the end of the string matches the
        one at the beginnin (ie. somehow multiple fields have not been read in)
        2. the number of rows of the data for this field simlib matches the
        number of observations recorded in the metadata as NOBS

        Parameters
        ----------
        validate_string : string, mandatory
            footer obtained by splitting the simlib corresponding to the field
            usually of the form 
        """

        val = eval(validate_string.split()[-1])
        if int(self.meta['LIBID']) != val:
            print('LIBID value at beginning: ', self.meta['LIBID'])
            print('LIBID value at the end', val)
            raise ValueError('the LIBID values do not match')

        if len(self.data) != self.meta['NOBS']:
            print('NOBS :', self.meta['NOBS'])
            print('len(data) :', len(self.data))
            raise ValueError('the number of observations recorded does not'
                             'match size of data')

    @staticmethod
    def split_simlibString(simlibString):
        '''
        split the string corresponding to a simlib file into header, footer,
        and data pieces

        Parameters
        ----------
        simlibString : string
        '''
        lst = simlibString.split('MAG')
        header = lst[0]
        data, val = lst[1].split('END_LIBID')
        index = data.index('\n')
        return header, data[index+1:], val

    @staticmethod
    def simlibdata(data):
        '''
        manipulate string in the simlibstring to form pandas DataFrame object


        Parameters
        ----------

        data : data
        '''
        fhandle = StringIO(data)
        df = pd.read_csv(fhandle, delimiter="\s+",
                         names=['trash', 'MJD', 'IDEXPT', 'FLT', 'GAIN',
                                'NOISE', 'SKYSIG', 'PSF1', 'PSF2',
                                'PSFRatio', 'ZPTAVG', 'ZPTERR', 'MAG'])
        del df['trash']
        return df

    @staticmethod
    def split_header(header):
        '''
        split header string into metadata and field names

        Parameters
        ----------

        header : header
        '''

        lines = header.split('\n')
        header_metadata = []
        header_fields = []
        for line in lines:
            if line.startswith('#'):
                header_fields.append(line[1:])
            else:
                header_metadata += line.split()
        return header_metadata, header_fields

    @staticmethod
    def libid_metadata(header_metadata):
        '''
        parse header metadata string into a dictionary

        Parameters
        ----------
        header_metadata : header
        '''

        # Even index values 0, 2, 4 are keys
        # remove ':' char at end
        keys = list(map(lambda x: x[:-1], header_metadata[0::2]))

        # odd index values are floats or ints
        vals = list(map(eval, header_metadata[1::2]))

        return dict(zip(keys, vals))

