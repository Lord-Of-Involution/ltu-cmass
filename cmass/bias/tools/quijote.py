

# Quijote
import astropy.units as apu
from astropy.coordinates import SkyCoord
import scipy.stats
from scipy.optimize import minimize
import numpy as np
import os


class FoF_catalog:
    def __init__(self, basedir, snapnum, long_ids=False, swap=False,
                 SFR=False, read_IDs=True, prefix='/groups_'):

        if long_ids:
            format = np.uint64
        else:
            format = np.uint32

        exts = ('000'+str(snapnum))[-3:]

        #################  READ TAB FILES #################
        fnb, skip, Final = 0, 0, False
        dt1 = np.dtype((np.float32, 3))
        dt2 = np.dtype((np.float32, 6))
        prefix = basedir + prefix + exts + "/group_tab_" + exts + "."
        while not (Final):
            f = open(prefix+str(fnb), 'rb')
            self.Ngroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNgroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.Nids = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNids = np.fromfile(f, dtype=np.uint64, count=1)[0]
            self.Nfiles = np.fromfile(f, dtype=np.uint32, count=1)[0]

            TNG, NG = self.TotNgroups, self.Ngroups
            if fnb == 0:
                self.GroupLen = np.empty(TNG, dtype=np.int32)
                self.GroupOffset = np.empty(TNG, dtype=np.int32)
                self.GroupMass = np.empty(TNG, dtype=np.float32)
                self.GroupPos = np.empty(TNG, dtype=dt1)
                self.GroupVel = np.empty(TNG, dtype=dt1)
                self.GroupTLen = np.empty(TNG, dtype=dt2)
                self.GroupTMass = np.empty(TNG, dtype=dt2)
                if SFR:
                    self.GroupSFR = np.empty(TNG, dtype=np.float32)

            if NG > 0:
                locs = slice(skip, skip+NG)
                self.GroupLen[locs] = np.fromfile(f, dtype=np.int32, count=NG)
                self.GroupOffset[locs] = np.fromfile(
                    f, dtype=np.int32, count=NG)
                self.GroupMass[locs] = np.fromfile(
                    f, dtype=np.float32, count=NG)
                self.GroupPos[locs] = np.fromfile(f, dtype=dt1, count=NG)
                self.GroupVel[locs] = np.fromfile(f, dtype=dt1, count=NG)
                self.GroupTLen[locs] = np.fromfile(f, dtype=dt2, count=NG)
                self.GroupTMass[locs] = np.fromfile(f, dtype=dt2, count=NG)
                if SFR:
                    self.GroupSFR[locs] = np.fromfile(
                        f, dtype=np.float32, count=NG)
                skip += NG

                if swap:
                    self.GroupLen.byteswap(True)
                    self.GroupOffset.byteswap(True)
                    self.GroupMass.byteswap(True)
                    self.GroupPos.byteswap(True)
                    self.GroupVel.byteswap(True)
                    self.GroupTLen.byteswap(True)
                    self.GroupTMass.byteswap(True)
                    if SFR:
                        self.GroupSFR.byteswap(True)

            curpos = f.tell()
            f.seek(0, os.SEEK_END)
            if curpos != f.tell():
                raise Exception(
                    "Warning: finished reading before EOF for tab file", fnb)
            f.close()
            fnb += 1
            if fnb == self.Nfiles:
                Final = True

        #################  READ IDS FILES #################
        if read_IDs:

            fnb, skip = 0, 0
            Final = False
            while not (Final):
                fname = basedir+"/groups_" + exts + \
                    "/group_ids_"+exts + "."+str(fnb)
                f = open(fname, 'rb')
                Ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
                TotNgroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
                Nids = np.fromfile(f, dtype=np.uint32, count=1)[0]
                TotNids = np.fromfile(f, dtype=np.uint64, count=1)[0]
                Nfiles = np.fromfile(f, dtype=np.uint32, count=1)[0]
                Send_offset = np.fromfile(f, dtype=np.uint32, count=1)[0]
                if fnb == 0:
                    self.GroupIDs = np.zeros(dtype=format, shape=TotNids)
                if Ngroups > 0:
                    if long_ids:
                        IDs = np.fromfile(f, dtype=np.uint64, count=Nids)
                    else:
                        IDs = np.fromfile(f, dtype=np.uint32, count=Nids)
                    if swap:
                        IDs = IDs.byteswap(True)
                    self.GroupIDs[skip:skip+Nids] = IDs[:]
                    skip += Nids
                curpos = f.tell()
                f.seek(0, os.SEEK_END)
                if curpos != f.tell():
                    raise Exception(
                        "Warning: finished reading before EOF for IDs file", fnb)
                f.close()
                fnb += 1
                if fnb == Nfiles:
                    Final = True


def load_quijote_halos(snapdir):
    FoF = FoF_catalog(snapdir, snapnum=4,  # 4 for z=0
                      long_ids=False,
                      swap=False, SFR=False, read_IDs=False)

    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3  # Halo positions in Mpc/h
    mass = FoF.GroupMass*1e10  # Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+0)  # Halo peculiar velocities in km/s
    Npart = FoF.GroupLen  # Number of CDM particles in the halo

    return pos_h, mass, vel_h, Npart
