"""Declared benchmark sets. Declared before running; failures stay in the results."""

# netlib.org/lp/data: every feasible LP of the collection (98 models).
NETLIB = """25fv47 80bau3b adlittle afiro agg agg2 agg3 bandm beaconfd blend bnl1 bnl2 boeing1 boeing2
bore3d brandy capri cycle czprob d2q06c d6cube degen2 degen3 dfl001 e226 etamacro fffff800 finnis
fit1d fit1p fit2d fit2p forplan ganges gfrd-pnc greenbea greenbeb grow15 grow22 grow7 israel kb2
lotfi maros maros-r7 modszk1 nesm perold pilot pilot.ja pilot.we pilot4 pilot87 pilotnov qap8 qap12
qap15 recipe sc105 sc205 sc50a sc50b scagr25 scagr7 scfxm1 scfxm2 scfxm3 scorpion scrs8 scsd1 scsd6
scsd8 sctap1 sctap2 sctap3 seba share1b share2b shell ship04l ship04s ship08l ship08s ship12l ship12s
sierra stair standata standgub standmps stocfor1 stocfor2 stocfor3 truss tuff vtp.base wood1p woodw""".split()

# netlib.org/lp/infeas: the infeasible collection (29 models).
NETLIB_INFEAS = """bgdbg1 bgetam bgindy bgprtr box1 ceria3d chemcom cplex1 cplex2 ex72a ex73a forest6 galenet
gosh gran greenbea-infeas itest2 itest6 klein1 klein2 klein3 mondou2 pang pilot4i qual reactor refinery vol1
woodinfe""".split()

KENNINGTON = """ken-07 ken-11 ken-13 ken-18 osa-07 osa-14 osa-30 osa-60 pds-02 pds-06 pds-10 pds-20 cre-a
cre-b cre-c cre-d""".split()

# large Mittelmann LPs used for the GPU runs
GATE = "qap15 nug08-3rd savsched1 pds-100 rmine15 Linf_520c cont1".split()

# Maros-Meszaros convex QP collection (138 models; QPS files from YimingYAN/QP-Test-Problems)
MAROS = """AUG2D AUG2DC AUG2DCQP AUG2DQP AUG3D AUG3DC AUG3DCQP AUG3DQP BOYD1 BOYD2 CONT-050 CONT-100
CONT-101 CONT-200 CONT-201 CONT-300 CVXQP1_L CVXQP1_M CVXQP1_S CVXQP2_L CVXQP2_M CVXQP2_S CVXQP3_L CVXQP3_M
CVXQP3_S DPKLO1 DTOC3 DUAL1 DUAL2 DUAL3 DUAL4 DUALC1 DUALC2 DUALC5 DUALC8 EXDATA
GENHS28 GOULDQP2 GOULDQP3 HS118 HS21 HS268 HS35 HS35MOD HS51 HS52 HS53 HS76
HUES-MOD HUESTIS KSIP LASER LISWET1 LISWET10 LISWET11 LISWET12 LISWET2 LISWET3 LISWET4 LISWET5
LISWET6 LISWET7 LISWET8 LISWET9 LOTSCHD MOSARQP1 MOSARQP2 POWELL20 PRIMAL1 PRIMAL2 PRIMAL3 PRIMAL4
PRIMALC1 PRIMALC2 PRIMALC5 PRIMALC8 Q25FV47 QADLITTL QAFIRO QBANDM QBEACONF QBORE3D QBRANDY QCAPRI
QE226 QETAMACR QFFFFF80 QFORPLAN QGFRDXPN QGROW15 QGROW22 QGROW7 QISRAEL QPCBLEND QPCBOEI1 QPCBOEI2
QPCSTAIR QPILOTNO QPTEST QRECIPE QSC205 QSCAGR25 QSCAGR7 QSCFXM1 QSCFXM2 QSCFXM3 QSCORPIO QSCRS8
QSCSD1 QSCSD6 QSCSD8 QSCTAP1 QSCTAP2 QSCTAP3 QSEBA QSHARE1B QSHARE2B QSHELL QSHIP04L QSHIP04S
QSHIP08L QSHIP08S QSHIP12L QSHIP12S QSIERRA QSTAIR QSTANDAT S268 STADAT1 STADAT2 STADAT3 STCQP1
STCQP2 TAME UBH1 VALUES YAO ZECEVIC2""".split()

SMALL = "afiro sc50a sc50b sc105 kb2 adlittle blend stocfor1 share2b recipe".split()

SETS = {"netlib": NETLIB, "netlib_infeas": NETLIB_INFEAS, "kennington": KENNINGTON, "gate": GATE,
        "small": SMALL, "maros": MAROS}
assert len(MAROS) == 138 and len(NETLIB) == 98 and len(NETLIB_INFEAS) == 29 and len(KENNINGTON) == 16
