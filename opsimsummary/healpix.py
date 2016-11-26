"""
Class for associating Healpixels with OpSim Pointings. An example of usage can
be found in `examples/ObsHistIDsForTile`
"""
from __future__ import print_function, absolute_import, division
import numpy as np
import healpy as hp
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import repeat 
import sqlite3
from .opsim_out import OpSimOutput
import subprocess
from datetime import datetime

__all__  = ['addVec', 'HealPixelizedOpSim']

def addVec(df, raCol='ditheredRA', decCol='ditheredDec'):
    """
    Add a column of vectors to the dataFrame df

    Parameters
    ----------
    df : `pd.DataFrame`
        dataframe with two columns raCol and decCol having ra,
        dec in radians.
    raCol : string, optional, defaults to 'ditheredRA'
        column name for the column which has ra values
    decCol : string, optional, defaults to 'ditheredDec'
        column name for the column which has dec values
    """
    thetas  = - df[decCol] + np.pi /2.
    phis = df[raCol]
    df['vec'] = list(hp.ang2vec(thetas, phis))

class HealPixelizedOpSim(object):
    """
    Class to associate opsim pointings represented as records indexed by an
    integer variable obsHistID, with a set of healpixel tileIds. This class
    computes the (maximal set) of healpixel IDs associated with a certain
    pointing, as also the set of pointings associated with a healpixel ID.


    Parameters
    ----------
    opsimDF : `pd.DataFrame`, mandatory
        a dataframe representing the OpSim records of interest. The mandatory
        columns are an index column (obsHistID), raCol (specified by raCol), 
        dec Col (specified as decCol)
    raCol : string, defaults to 'ditheredRa', col should have units of radians
        column name for column of ras to use 
    decCol : string, defaults to 'ditheredDec', col should have units of radians
        column name for column of dec to use
    NSIDE : integer, `healpy.NSIDE`
        `NSIDE` for healpix giving rise to 12NSIDE **2 pixels on the sphere
    vecColName : string, optional, defaults to 'vec'
        column name where 3D vectors are computed corresponding to the angles
        of the pointing direction of the OpSim record.
    fieldRadius : float, optional, defaults to 1.75, units is degrees
        radius of the field in degrees
    inclusive : bool, optional, defaults to True
        If False, healpixels whose centers fall within the fieldRadius of an
        OpSim pointing are associated with the pointing. This misses those
        Healpixels that partially overlap the pointing. If inclusive = True, an
        approximate method is used to include such missing pixels, by
        associating a healpixel to the pointing if the healpixel includes a pixel
        at a resolution NSIDE*fact which has a center falling within the
        fieldRadius. 
    fact : int, optional, defaults to 4
        Determines the effective NSIDE for associating healpixels to OpSim
        records, as described in the documentaiton for `inclusive`

    source : string, optional, defaults to None
        if not None, used to record the absolute path or name of the OpSim
        output database on which this object was based
    Methods
    -------
    """

    def __init__(self, opsimDF, raCol='ditheredRA', decCol='ditheredDec',
                 NSIDE=1, fact=4, inclusive=True, nest=True,
                 vecColName='vec',  fieldRadius=1.75, source=None):

        self.raCol = raCol
        self.decCol = decCol
        self.opsimdf = opsimDF.reset_index()
        self.cols = opsimDF.columns
        self.vecColName = vecColName
        self.validateDF()
        self._fieldRadius = np.radians(fieldRadius)

        if vecColName not in self.cols:
            addVec(self.opsimdf, raCol=self.raCol, decCol=self.decCol)
        self.nside = NSIDE
        self._rowdata = None
        self._coldata = None
        self._spmat = None
        self.inclusive  = inclusive
        self.fact  = fact
        self.nest = nest
        self.source = source

    @classmethod
    def fromOpSimDB(cls, opSimDBpath, subset='combined', propIDs=None,
                    NSIDE=256, raCol='ditheredRA', decCol='ditheredDec',
                    inclusive=True, fact=4, nest=True,
                    fieldRadius=1.75,  vecColName='vec'):
        """
        Parameters
        ----------
        opSimDBpath : 
        subset :
        propIDs :
        raCol :
        decCol :
        NSIDE :
        fieldRadius :
        fact :
        inclusive : 
        nest :

        Returns
        -------
        instance of the class
        """
        tableNames = ('Summary', 'Proposal')
        subset = subset
        propIDs = propIDs
        dbName = opSimDBpath

        opsimout = OpSimOutput.fromOpSimDB(opSimDBpath, subset=subset,
                                           tableNames=tableNames,
                                           propIDs=propIDs)
        summary = opsimout.summary 
        return cls(opsimDF=summary, raCol=raCol, decCol=decCol,
                 NSIDE=NSIDE, vecColName=vecColName, nest=nest, source=dbName,
                 fieldRadius=fieldRadius, fact=fact, inclusive=inclusive)


    
        
        
    @classmethod
    def fromOpSimHDF(cls, opSimHDF, subset='combined', propIDs=None,
                    NSIDE=256, raCol='ditheredRA', decCol='ditheredDec',
                    fieldRadius=1.75,  vecColName='vec'):
        """
        """
        tableNames = ('Summary', 'Proposal')
        subset = subset
        propIDs = propIDs
        opsimHDF = opsimHDF
        
        opsimout = OpSimOutput.fromOpSimHDF(opSimHDF, subset=subset,
                                           tableNames=tableNames,
                                           propIDs=propIDs)
        summary = opsimout.summary 
        return cls(opsimDF=summary, raCol=raCol, decCol=decCol,
                 NSIDE=NSIDE, vecColName=vecColName,
                 fieldRadius=fieldRadius)

        
    def obsHistIdsForTile(self, tileID):
        """
        return a `np.ndarray` of obsHistID values that intersect with the
        healpix tileID.
        """
        inds = self.sparseMat.getcol(tileID).nonzero()[0]
        return self.opsimdf.ix[inds, 'obsHistID'].values

    @property
    def sparseMat(self):
        """
        Sparse Matrix representation of the association between the obsHistIDs
        and healpix tileIDs.
        """
        if self._spmat is None:
            shape=(len(self.opsimdf), hp.nside2npix(self.nside))
            # numpy ones like number of intersections
            ones_int = np.ones(len(self.rowdata))
            self._spmat = csr_matrix((ones_int, (self.rowdata, self.coldata)),
                                     shape=shape)
        return self._spmat 

    @property
    def rowdata(self):
        if self._rowdata is None:
            self.doPreCalcs()
        return np.asarray(self._rowdata)


    @property
    def coldata(self):
        if self._coldata is None:
            self.doPreCalcs()
        return self._coldata

    def write_metaData_Table(self, dbName, indexed, version=None, hostname=None):
        """
        write out the metadata table to the sqlite database `dbName`. The
        columns of the table are hostname, CodeVersion, NSIDE, fact, indexed,
        inclusive, timestamp

        Parameters
        ----------
        dbName : string, mandatory
            absolute path to the database
        indexed : Bool, mandatory
            information on whether the main table `simlib` is indexed or not
        version : string, optional, defaults to None
            if None, this is set to 'Unknonwn'. The recommended way to use this
            is by supplying this variable to writeToDB, after finding it using
            `opsimsummary.__VERSION__`
        hostname : string, optional, defaults to None
            if None, the hostname is derived by using a subprocess call to the
            unix commandline `hostname`. Else, can be supplied.
        """
        conn = sqlite3.Connection(dbName)
        cur = conn.cursor()

        if version is None:
            version = 'UnKnown'

        if hostname is None:
            proc = subprocess.Popen('hostname', stdout=subprocess.PIPE)
            hostname, err = proc.communicate()
        

        # TimeStamp
        mytime = datetime.now() 
        timestamp = 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(mytime)

        # source
        source = self.source

        cur.execute('CREATE TABLE metadata ('
                                            'hostname varchar(100),'
                                            'source varchar(100),'
                                            'CodeVersion varchar(100),'
                                            'NSIDE int,'
                                            'fact int,'
                                            'inclusive varchar(10),'
                                            'indexed varchar(1),'
                                            'timestamp varchar(30))')
        insertStatement = 'INSERT INTO metadata '
        insertStatement += '(hostname, source, CodeVersion, NSIDE, fact, inclusive,'
        insertStatement += ' indexed, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?) '

        x = '{0},{1},{2},{3},{4},{5},{6},{7}'.format(hostname, source, version,
                                                       self.nside, self.fact,
                                                       self.inclusive, indexed,
                                                       timestamp)
        vals = tuple(x.split(','))
        print(insertStatement)
        cur.execute(insertStatement, vals)
        conn.commit()                                            
        return




    def writeToDB(self, dbName, verbose=False, indexed=True, version=None,
                  hostname=None):
        """
        Write two tables to a SQLITE database. The first table is called Simlib
        and records association of obsHistIDs and Healpix TileIDs in a two
        column table. The second table is called metadata which records the
        provenance information in the table. The Simlib table may or may not be
        indexed
        Parameters
        ----------
        dbName : string, mandatory
            absolute path to the location of the database to be written
        verbose : Bool, optional, defaults to False
            determines the amount of output on committing records to the
            database
        indexed : Bool, optional, defaults to True
            if True, indexes both the columns of `Simlib` table
        version : string, optional, defaults to None
            a string that may be supplied to enumerate version numbers of the
            code. The recommended way to do this is via the use of
            `opsimsummary.__VERSION__`
        hostname : string, optional, defaults to None
            The hostname is used to supply the parameter in the metadata table.
            If None, that should be found by using the *NIX `hostname` command.

        .. notes : It is assumed that the file does not exist but the directory
        does.
        """
        rowdata = self.rowdata
        coldata = self.coldata
        if verbose:
            print(len(rowdata), len(coldata))
        obsHistIDs = self.opsimdf.ix[rowdata, 'obsHistID'].values

        conn = sqlite3.Connection(dbName)
        cur = conn.cursor()
        cur.execute('CREATE TABLE simlib (ipix int, obsHistId int)')
        for i in range(len(rowdata)):
            cur.execute('INSERT INTO simlib VALUES'
                        '({1}, {0})'.format(obsHistIDs[i], coldata[i]))
            if i % 100000 == 0:
                conn.commit()
                if verbose:
                    print('committed 100000 records to db')
        conn.commit()
        print('Committed the table to disk\n')
        # create index
        if indexed:
            print('Createing ipix index\n')
            cur.execute('CREATE INDEX {ix} on {tn}({cn})'\
                        .format(ix='ipix_ind', tn='simlib', cn='ipix'))
            print('Createing obsHistID index\n')
            cur.execute('CREATE INDEX {ix} on {tn}({cn})'\
                        .format(ix='obshistid_ind', tn='simlib', cn='obsHistId'))
        else:
            print('Not creating index \n')


        conn.close()
        # Write metadata table
        self.write_metaData_Table(dbName=dbName, indexed=indexed,
                                  version=version, hostname=hostname) 
        
    def doPreCalcs(self):
        """
        Perform the precalculations necessary to set up the sparse matrix.
        """
        self.opsimdf['hids'] = [hp.query_disc(self.nside, vec, self._fieldRadius,
                                              inclusive=self.inclusive,
                                              fact=self.fact,
                                              nest=self.nest)
                                for vec in self.opsimdf[self.vecColName]]
        lens = map(len, self.opsimdf.hids.values)
        rowdata = []
        _ = list(rowdata.extend(repeat(i, lens[i]))
                 for i in xrange(len(self.opsimdf)))
        coldata = np.concatenate(self.opsimdf.hids.values)
        self._rowdata = rowdata
        self._coldata = coldata

    def validateDF(self):
        if self.raCol not in self.cols:
            raise ValueError('raCol {} not in OpSim cols'.format(self.raCol))
        if self.decCol not in self.cols:
            raise ValueError('decCol {} not in OpSim cols'.format(self.decCol))
        return

