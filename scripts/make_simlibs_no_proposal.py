#!/usr/bin/env python
"""
Script to make Simlib files from OpSim DataBases (supported versions include
OpSim v3, OpSim v4, AltSched outputs.
    To get usage : python make_simlibs.py -h
"""
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import healpy as hp
import opsimsummary as oss
from opsimsummary import Simlibs, OpSimOutput
import sys
import time
import datetime
import subprocess


def write_genericSimlib(simlibFilename, 
                        summary, 
                        minVisits, 
                        maxVisits, 
                        numFields,
                        footprint_hids=None,
                        footprint_hids_fname=None,
                        vetoed_hids=None,
                        author_name=None,
                        mapFile=None,
                        rng=np.random.RandomState(0),
                        mwebv=0., 
                        raCol='ditheredRA', 
                        decCol='ditheredDec',
                        angleUnit='degrees', 
                        opsimversion='fbsv2',
                        indexCol='obsHistID', 
                        nside=256, 
                        fieldType='DDF',
                        #opsimoutput='minion_1016_sqlite.db',
                        #opsim_output='opsim_output',
                        script_name=None,
                        surveypix_file=None):
    """
    Write out simlibs from a summary dataFrame

    Parameters
    ----------
    simlibFilename : string
        absolute path to simlib file to be written out
    summary : `pd.dataFrame`
        summary of observations
    minVisits : int
        minimum number of visits, such that fields with lower numbers of visits
        over the survey are not considered.
    maxVisits : int
        maximum number of visits, such that fields with higher numbers of visits
        over the survey are not considered.
    numFields : int
        number of sample fields requested in the simlib file. If this number
        exceeds the number of fields satisfying the number of visits, all of the
        fields satisfying the constraints are included in the simlib
    author_name : string, defaults to None
        Author name. If not provided, assumed to be None. The author name will be replaced by 
        user login.
    vetoed_hids : set of integers, defaults to `None`
       if not `None`, set of integers representing healpix Ids in the same
       (`nest`|`ring`) which should not be used. If `None`, ignore.
    opsim_output: str
        name of the opsim_output 'opsim_output'
    script_name : 
    surveypix_file : abspath to surveypix_file, defaults to None
        If not None, uses this file to find the libids to simulate. Should
        have an ordered dataframe of ra, dec, simlibIds

    Returns
    -------
    surveypix : dataframe
        healpixels in our selected footprint
    surveydf : dataframe
        selected visits with information needed for simlib files


    Notes
    -----
        Aside from the returns, this function also writes out the simlib files, and the surveypix_file if it was 
        constructed in the method. It also returns a set of logs to stdout. 
    """
    print('========================')
    print(mapFile, surveypix_file)

    print('========================')

    opsimsummary_version = oss.__version__

    minMJD = summary.expMJD.min()
    maxMJD = summary.expMJD.max()

    simlibs = Simlibs(summary,
                      usePointingTree=True,
                      raCol=raCol,
                      decCol=decCol,
                      angleUnit=angleUnit,
                      indexCol=indexCol,
                      opsimversion=opsimversion)

    # This is a dataframe with healpix Ids that get between minVisits
    # and maxVisits

    surveydf = simlibs.observedVisitsinRegion(minVisits=minVisits,
                                              maxVisits=maxVisits,
                                              writeFile=False,
                                              nside=nside)

    # In SNANA, we do a simulation by sampling observed visits
    # The sampling is done by choosing a spatial samples of the footprint
    # which must be representative of the footprint, rather than things like edges.
    # This selected footprint is described as a set of healpixels and is described by
    # indices called hids in surveypix


    # First collect hids of the footprint if available

    ## use sequence if provided
    if footprint_hids is not None:
        hids = set(footprint_hids)
        print('Using passed sequence for hids\n')
    elif footprint_hids_fname is not None:
        surveyPix = pd.read_csv(footprint_hids_fname)
        hids = set(surveyPix.index.values)
        print('Obtaining hids from provided surveyPix file\n') 
    elif summary is not None:
        hids = set(surveydf.query('numVisits >= @minVisits').index.values)
        print('Constructing hids from simlibs from summary\n')
    else:
        raise ValueError('It does not make sense to not have a summary file \n')


    if vetoed_hids is not None:
        vetoed_hids = set(vetoed_hids)
        avail = hids - vetoed_hids
    else:
        avail = hids

    surveydf = surveydf.loc[list(avail)].query('numVisits >= @minVisits ')
    surveyPix = simlibs.get_surveyPix(surveydf, numFields=numFields, rng=rng)
    totalfields = len(surveydf)


    ####    # The surveypix may be constructed or read in from a file : `surveypix_file` 
    ####    if surveypix_file is None:
    ####        # vetoed hids may describe parts of the footprint that are not part of the survey, e.g Deep Drilling
    ####        # Fields when you want to study Wide Fast Deep.
    ####        if vetoed_hids is not None:
    ####            hids = set(surveydf.index.values)
    ####            selected = hids - vetoed_hids
    ####            surveydf = surveydf.loc[selected]
    ####        totalfields = len(surveydf)
    ####        surveyPix = simlibs.get_surveyPix(surveydf, numFields=numFields, rng=rng)
    ####    else:
    ####        print('Reading in surveypix file\n')
    ####        print('You should only be using this if you are studying ToO proposals\n')
    ####        surveyPix = pd.read_csv(surveypix_file)
    ####        totalfields = len(surveyPix)

    fields = simlibs.simlibs_for_fields(surveyPix, mwebv=mwebv)

    # Area of our selected footprint in sq. deg 
    area = hp.nside2pixarea(nside, degrees=True) * np.float64(totalfields)

    # Area of our selected footprint in solid angle (Used by SNANA for simulation)
    solidangle = hp.nside2pixarea(nside, degrees=False) * np.float64(totalfields)

    print('Going to write simlib file {0} for opsim output\n')
    # This is not very reliable
    if script_name is None:
        script_name = 'OpSimSummary/scripts/make_simlibs.py'
    dt = datetime.datetime.now()
    ts = dt.isoformat()


    # Author name. Will be entered in SNANA Simlib documentation. 
    if author_name is None:
        author_name = os.getlogin()

    ## git information
    try :
        label = subprocess.check_output(['git', 'describe', '--dirty']).strip().decode('utf-8')
        # Assume that there will be git versions before and after 2.22
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        if label.endswith('dirty'):
            git_sha = label.split('-dirty')[0]
            dirty = 'dirty'
        else:
            git_sha = label
            dirty = ''

        git_info = f'The script was part of a {dirty} git repository in the branch {branch} and commit SHA {git_sha} and was labelled with a version : {oss.__version__}'
    except:
        git_info = 'The script was not detected to be a part of a git repository'


    # OpSimSummary version information 
    version = oss.__version__   # RB: I think this line is not needed

    COMMAND = ' '.join(sys.argv)  # RK^2
    comment =  f'DOCUMENTATION:\n'
    comment += f'    PURPOSE: simulate LSST based on mock opsim version {opsim_output}\n'
    comment += f'    INTENT:   Nominal\n'
    comment += f'    USAGE_KEY: SIMLIB_FILE\n'
    comment += f'    USAGE_CODE: snlc_sim.exe\n'
    comment += f'    VALIDATION_SCIENCE: \n'

    # xxx mark comment += f'    FIELD: {fieldType}\n' # obsolete logic 
    comment += f'    FIELD: WFD+DDF\n'  # R.Kessler Jul 2024: no more WFD or DDF only options

    comment += f'    NOTES: \n'
    comment += f'        PARAMS MINMJD: {minMJD:.4f}\n'
    comment += f'        PARAMS MAXMJD: {maxMJD:.4f}\n'
    comment += f'        PARAMS TOTAL_AREA: {area:.3f}\n'
    comment += f'        PARAMS SOLID_ANGLE: {solidangle:.3f}\n'
    comment += f'        WARNING_NOTICE: DDF approximately determined by nobs>=1100\n'
    comment += f'        COMMAND: {COMMAND}\n'
    comment += f'    VERSIONS:\n'
    comment += f'    - DATE : {dt.strftime(format="%y-%m-%d")}\n'
    comment += f'    AUTHORS : {author_name}, OpSimSummary version {oss.__version__}\n'
    comment += 'DOCUMENTATION_END:\n'

    doc = comment

    # Additional commments which can be used as keys, or as additional documentation
    # SKIP
    ####    comment = 'COMMENT: Total area corresponding to this simlib is {0:.1f} sq degrees or a solid angle of {1:4f} \n'.format(area, solidangle) 
    ####    comment += 'COMMENT: This is a simlib corresponding to {0} in the OpSim Output {1} using the script {3} in version {2} at time {4}. {5}\n'.format(
    ####            fieldType,
    ####            opsimoutput,
    ####            opsimsummary_version,
    ####            script_name,
    ####            ts,
    ####            git_info)
    ####    comment += '\nCOMMENT: PARAMS MINMJD: {:.4f}\n'.format(minMJD)
    ####    comment += 'COMMENT: PARAMS MAXMJD: {:.4f}\n'.format(maxMJD)
    ####    comment += 'COMMENT: PARAMS TOTAL_AREA: {:.3f}\n'.format(area)
    ####    comment += 'COMMENT: PARAMS SOLID_ANGLE: {:.3f}\n'.format(solidangle)

    simlibs.writeSimlib(simlibFilename,
                        fields,
                        mwebv=mwebv,
                        doc=doc,
                        comments='\n',
                        numLibId=numFields)
    surveyPix = surveyPix.reset_index().query('simlibId > -1').set_index('simlibId')
    surveyPix = surveyPix.reset_index().sort_values(by='simlibId').set_index('simlibId')

    # Write out the surveyPix
    mapFile = 'testing' + fieldType
    surveyPix.to_csv(mapFile)
    return surveyPix, surveydf

if __name__ == '__main__':
    parser = ArgumentParser(description='write out simlibs from an OpSim Database')
    parser.add_argument('--data_root', help='absolute path to data directory containing dither files and opsim database, defaults to "/"',
                        default='/')
    parser.add_argument('--dbname', help='path to sqlite database output from OpSim relative to data_root, defaults to `None`, which will crash', default=None)
    parser.add_argument('--opsim_output', help='Optionally required for non-standard/non-Rubin opsim output developments: name of the opsim output being used, will be obtained from filename if skipped', default=None, type=str)
    parser.add_argument('--no_write_ddf_simlib', help='Whether to write out DDF simlib',
                        dest='write_ddf_simlib', action='store_false')
    parser.add_argument('--no_get_ddf_pixels', help='Whether to obtain ddf pixels from opsim or use null set',
                        dest='No_get_ddf_pixels', action='store_false')
    parser.add_argument('--no_write_wfd_simlib', help='Whether to write out WFD simlib, defaults to writing it out',
                        dest='write_wfd_simlib', action='store_false')
    parser.add_argument('--opsimversion', help='version of opsim used lsstv3|lsstv4|fbsv2, defaults to fbsv2',
                        default='fbsv2')
    parser.add_argument('--summaryTableName', help='name of Summary Table Summary|SummaryAllProps, defaults to "Summary"',
                        default='Summary')
    parser.add_argument('--ddf_surveypix_file', help='absolute path to DDF surveypix_file, defaults to `None`',
                        default=None)
    parser.add_argument('--wfd_surveypix_file', help='absolute path to WFD surveypix_file, defaults to `None`',
                        default=None)
    parser.add_argument('--ddf_simlibfilename', help='absolute path to DDF simlib file to write out, defaults to `None`',
                        default=None)
    parser.add_argument('--wfd_simlibfilename', help='absolute path to WFD simlib file to write out, defaults to `None`',
                        default=None)
    parser.add_argument('--numFields_DDF', help='number of locations in DDF where simlib fields are located, defaults to 133',
                        default=133, type=int)
    parser.add_argument('--numFields_WFD', help='number of locations in DDF where simlib fields are located, defaults to 50000',
                        default=50000, type=int)
    parser.add_argument('--filterNull', help='if added, then the summary table of the OpSim file will be filtered of rows that appear to have null values',
                        dest='filt_Null', action='store_true')
    parser.add_argument('--authorName', help='Name of the author to be recorded in documentation. If skipped, the login will be used instead', type=str, default=None)

    print("read in command line options and figuring out what to do\n")
    print("---------------------------------------------------------\n")
    print("We are using opsimsummary version {0}, the library is located at {1}".format(oss.__version__, oss.__file__))
    print("The path is {}".format(sys.path))
    print("The python version is {}".format(sys.version))
    print("---------------------------------------------------------\n")
    sys.stdout.flush()
    args = parser.parse_args()

    print("---------------------------------------------------------\n")
    print("Obtain arguments\n")
    print("---------------------------------------------------------\n")
    if args.dbname is None:
        print('You have to supply `dbname` as the absolute path to the OpSim Output. \n')
        print('Alternatively `dbname` can also be a filename, relative to `data_root` which must be specified as well. \n')
        raise ValueError('`dbname` cannot be `None`')

    data_root = args.data_root
    author_name = args.authorName
    dbname = os.path.join(data_root, args.dbname)
    basename = dbname.split('/')[-1].split('.db')[0]

    opsim_output = args.opsim_output
    if opsim_output is None:
        opsim_output = basename

    filternulls = False
    if args.filt_Null :
        print('filtering the raw summary table, with this option, null values in `fiveSigmaDepth` will be skipped')
        print("We do not recommend using this option unless you are sure you want this\n")
        filternulls = True

    get_ddf_pixels = False
    print("get_ddf_pixels", get_ddf_pixels)


    print("\n\n Task: Obtaining pointing location \n")
    sys.stdout.flush()
    summaryTableName = args.summaryTableName
    opsimoutput = os.path.basename(dbname)
    opsimversion = args.opsimversion
    write_wfd_simlib = args.write_wfd_simlib
    write_ddf_simlib = False
    wfd_simlibfilename = args.wfd_simlibfilename
    ddf_simlibfilename = False

    if wfd_simlibfilename is None:
        wfd_simlibfilename = basename +'_all.simlib'
    availwfdFileName = basename + "_all_avail.csv"
    selectedwfdFileName = basename + "_all_sel.csv"
    print('output file names for all are {0}, {1}, {2}'.format(wfd_simlibfilename, availwfdFileName, selectedwfdFileName))

    numFields_WFD = args.numFields_WFD
    print(args)

    sys.stdout.flush()

    # find ddf healpixels if the `get_ddf_pixels` is True and 
    # store in ddf_hid
    # Else:  ddf_hid is an empty set
    get_ddf_pixels = False
    if get_ddf_pixels:
        print('Finding the DDF healpixels \n')
        # read in opsim with only DDF observations
        opsout_ddf = OpSimOutput.fromOpSimDB(dbname, opsimversion=opsimversion,
                                             subset='ddf',
                                             filterNull=filternulls)


        if len(opsout_ddf.summary) > 0:
            print("writing out ddf pixels\n")
            simlib_ddf = Simlibs(opsout_ddf.summary, opsimversion=opsimversion,
                                 usePointingTree=True)
            # These are the ddf healpix pixels
            ddf_hid = set(simlib_ddf.observedVisitsinRegion().index.values) 
    else:

        print("writing out null set of ddf pixels\n")
        ddf_hid = set([])
        print("written out null set of ddf pixels\n")


    print('There are {} pixels in the ddf fields'.format(len(ddf_hid)))
    # read the database into a `pd.DataFrame`
    tstart = time.time()
    print("\n\n Task: reading database {0} at time {1}. This can take a while ... ".format(dbname, tstart))
    sys.stdout.flush()

    # read with subset = 'combined', This should have WFD  + DDF pointings and reads in the entire opsim 
    opsout = OpSimOutput.fromOpSimDB(dbname,
                                     opsimversion=opsimversion,
                                     tableNames=(summaryTableName, 'Proposal'),
                                     subset='unique_all',
                                     filterNull=False,
                                     use_proposal_table=False)
    tend = time.time()
    print("finished reading database {0} at time {1}".format(dbname, tend))
    print("reading the db took {} minutes".format((tend-tstart)/60.0))
    sys.stdout.flush()

    # Summary table with this set.
    summary = opsout.summary
    script_name = os.path.abspath(__file__)
    print('========================')
    print(
            '\n dbname',  dbname, 
            '\n script_name', script_name, 
            '\n author_name', author_name,
            '\n opsim_output', opsim_output) 
    print('========================')


    write_ddf_simlib = False
    if write_ddf_simlib:
        print('\n\n Task: writing out simlib for DDF')
        # 133 random locations is similar density of locations in WFD.
        x, y = write_genericSimlib(simlibFilename=ddf_simlibfilename,
                                   summary=summary, 
                                   minVisits=500, maxVisits=None,
                                   numFields=numFields_DDF, 
                                   mapFile=None,
                                   fieldType='DDF', 
                                   # opsimoutput=dbname,
                                   script_name=script_name,
                                   author_name=author_name,
                                   # opsim_output=opsim_output,
                                   # surveypix_file=args.ddf_surveypix_file)
                                   footprint_hids=ddf_hid)
        print('Finished writing out simlib for DDF')
        print('\n\n Task: write mapping to csv')
        x.to_csv(selectedddfFileName)
        y.to_csv(availddfFileName)
        print('Finished writing mapping to csv')
        sys.stdout.flush()
    sys.stdout.flush()


    if write_wfd_simlib :
        print('\n\n Task: writing out simlib for WFD')
        sys.stdout.flush()
        # R.Kessler July 2024 increase maxVists from 100k to 300k, and add nside=512 arg
        #    (replace previous default of nside=256)
        x, y  = write_genericSimlib(simlibFilename=wfd_simlibfilename,
                                    summary=summary, minVisits=500, maxVisits=300000,
                                    numFields=numFields_WFD, mapFile='wfd_minion_1016_sqlite.csv',
                                    fieldType='WFD', # opsimoutput=dbname, 
                                    vetoed_hids=ddf_hid,
                                    nside=512,        # R.Kessler July 2024
                                    script_name=script_name)
        print('Finished writing out simlib for WFD')
        print('\n\n Task: write mapping to csv')
        x.to_csv(selectedwfdFileName)
        y.to_csv(availwfdFileName)
        print('Finished writing mapping to csv')
        sys.stdout.flush()
    print('finished job')
    sys.stdout.flush()
