from tespy.components.basics.cycle_closer import CycleCloser
from tespy.networks import Network
from tespy.components import (
    CycleCloser, Pipe, Pump, Valve, HeatExchangerSimple, Splitter, Merge
)
from tespy.connections import Connection

fluid_list = ['INCOMP::Water']
nw = Network(fluids=fluid_list)
nw.set_attr(T_unit='C', p_unit='bar', h_unit='kJ / kg')

# central heating plant
hs = HeatExchangerSimple('heat source')
cc = CycleCloser('cycle closer')
pu = Pump('feed pump')

# consumer
cons_0 = HeatExchangerSimple('consumer 0')
cons_1 = HeatExchangerSimple('consumer 1')
cons_2 = HeatExchangerSimple('consumer 2')
cons_3 = HeatExchangerSimple('consumer 3')


val_0 = Valve('control valve 0')
val_1 = Valve('control valve 1')
val_2 = Valve('control valve 2')
val_3 = Valve('control valve 3')

# pipes
pipe_feed_0 = Pipe('feed pipe 0')
pipe_feed_1 = Pipe('feed pipe 1')
pipe_feed_2 = Pipe('feed pipe 2')
pipe_feed_3 = Pipe('feed pipe 3')
pipe_feed_4 = Pipe('feed pipe 4')
pipe_feed_5 = Pipe('feed pipe 5')
pipe_feed_6 = Pipe('feed pipe 6')
pipe_return_0 = Pipe('return pipe 0')
pipe_return_1 = Pipe('return pipe 1')
pipe_return_2 = Pipe('return pipe 2')
pipe_return_3 = Pipe('return pipe 3')
pipe_return_4 = Pipe('return pipe 4')
pipe_return_5 = Pipe('return pipe 5')
pipe_return_6 = Pipe('return pipe 6')

# pipe splitters

sp_f_0 = Splitter("feed water splitter of source")
merg_r_0 = Merge("return water merger of source")
sp_f_1 = Splitter("feed water splitter of neighbors 1")
sp_f_2 = Splitter("feed water splitter of neighbors 2")
merg_r_1 = Merge("return water merger of neigbors 1")
merg_r_2 = Merge("return water merger of neigbors 2")


# connections
c0 = Connection(cc, "out1", hs, "in1", label="0")
c1 = Connection(hs, "out1", pu, "in1", label="1")
c2 = Connection(pu, "out1", pipe_feed_0, "in1", label="2")
c3 = Connection(pipe_feed_0, "out1", sp_f_0, "in1", label="3")

c4 = Connection(sp_f_0, "out1", pipe_feed_1, "in1", label="4")
c5 = Connection(sp_f_0, "out2", pipe_feed_4, "in1", label="5")

c6 = Connection(pipe_feed_1, "out1", sp_f_1, "in1", label="6")
c7 = Connection(sp_f_1, "out1", pipe_feed_2, "in1", label="7")
c8 = Connection(sp_f_1, "out2", pipe_feed_3, "in1", label="8")
c9 = Connection(pipe_feed_2, "out1", cons_0, "in1", label="9")
c10 = Connection(pipe_feed_3, "out1", cons_1, "in1", label="10")

c11 = Connection(pipe_feed_4, "out1", sp_f_2, "in1", label="11")
c12 = Connection(sp_f_2, "out1", pipe_feed_5, "in1", label="12")
c13 = Connection(sp_f_2, "out2", pipe_feed_6, "in1", label="13")
c14 = Connection(pipe_feed_5, "out1", cons_2, "in1", label="14")
c15 = Connection(pipe_feed_6, "out1", cons_3, "in1", label="15")

c16 = Connection(cons_1, "out1", val_1, "in1", label="16")
c17 = Connection(cons_0, "out1", val_0, "in1", label="17")

c18 = Connection(val_1, "out1", pipe_return_3, "in1", label="18")
c19 = Connection(val_0, "out1", pipe_return_2, "in1", label="19")
c20 = Connection(pipe_return_3, "out1", merg_r_1, "in1", label="20")
c21 = Connection(pipe_return_2, "out1", merg_r_1, "in2", label="21")
c22 = Connection(merg_r_1, "out1", pipe_return_1 , "in1", label="22")
c23 = Connection(pipe_return_1, "out1", merg_r_0 , "in1", label="23")

c24 = Connection(cons_3, "out1", val_3, "in1", label="24")
c25 = Connection(cons_2, "out1", val_2, "in1", label="25")

c26 = Connection(val_3, "out1", pipe_return_6, "in1", label="26")
c27 = Connection(val_2, "out1", pipe_return_5, "in1", label="27")
c28 = Connection(pipe_return_6, "out1", merg_r_2, "in1", label="28")
c29 = Connection(pipe_return_5, "out1", merg_r_2, "in2", label="29")
c30 = Connection(merg_r_2, "out1", pipe_return_4, "in1", label="30")
c31 = Connection(pipe_return_4, "out1", merg_r_0, "in2", label="31")
c32 = Connection(merg_r_0, "out1", pipe_return_0, "in1", label="32")
c33 = Connection(pipe_return_0, "out1", cc, "in1", label="33")


nw.add_conns(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
             c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33)

cons_0.set_attr(Q=-11100, pr=0.99)
cons_1.set_attr(Q=-5100, pr=0.99)
cons_2.set_attr(Q=-9100, pr=0.99)
cons_3.set_attr(Q=-6100, pr=0.99)
hs.set_attr(pr=1)
pu.set_attr(eta_s=0.7)

pipe_feed_0.set_attr(Q=-250, pr=0.99)
pipe_feed_1.set_attr(Q=-250, pr=0.99)
pipe_feed_2.set_attr(Q=-250, pr=0.99)
pipe_feed_3.set_attr(Q=-250, pr=0.99)
pipe_feed_4.set_attr(Q=-250, pr=0.99)
pipe_feed_5.set_attr(Q=-250, pr=0.99)
pipe_feed_6.set_attr(Q=-250, pr=0.99)

pipe_return_0.set_attr(Q=-200, pr=0.99)
pipe_return_1.set_attr(Q=-200, pr=0.99)
pipe_return_2.set_attr(Q=-200, pr=0.99)
pipe_return_3.set_attr(Q=-200, pr=0.99)
pipe_return_4.set_attr(Q=-200, pr=0.99)
pipe_return_5.set_attr(Q=-200, pr=0.99)
pipe_return_6.set_attr(Q=-200, pr=0.99)

c1.set_attr(T=90, p=3, fluid={'Water': 1})
c2.set_attr(p=5)
c20.set_attr(T=53)
c21.set_attr(T=59)
c28.set_attr(T=56)
c29.set_attr(T=61)


nw.set_attr(iterinfo=False)
nw.solve(mode="design")
nw.print_results()

pipe_feed_0.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=60,  # length in m
    D="var",  # diameter in m
)

pipe_feed_1.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=120,  # length in m
    D="var",  # diameter in m
)

pipe_feed_2.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=20,  # length in m
    D="var",  # diameter in m
)

pipe_feed_3.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=100,  # length in m
    D="var",  # diameter in m
)

pipe_feed_4.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=120,  # length in m
    D="var",  # diameter in m
)


pipe_feed_5.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=20,  # length in m
    D="var",  # diameter in m
)


pipe_feed_6.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=100,  # length in m
    D="var",  # diameter in m
)

pipe_return_0.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=60,  # length in m
    D="var",  # diameter in m
)

pipe_return_1.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=120,  # length in m
    D="var",  # diameter in m
)

pipe_return_2.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=20,  # length in m
    D="var",  # diameter in m
)

pipe_return_3.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=100,  # length in m
    D="var",  # diameter in m
)

pipe_return_4.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=120,  # length in m
    D="var",  # diameter in m
)

pipe_return_5.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=20,  # length in m
    D="var",  # diameter in m
)

pipe_return_6.set_attr(
    ks=0.0005,  # pipe's roughness in meters
    L=100,  # length in m
    D="var",  # diameter in m
)

nw.solve(mode="design")
nw.print_results()

pipe_feed_0.set_attr(D=pipe_feed_0.D.val, pr=None)
pipe_feed_1.set_attr(D=pipe_feed_1.D.val, pr=None)
pipe_feed_2.set_attr(D=pipe_feed_2.D.val, pr=None)
pipe_feed_3.set_attr(D=pipe_feed_3.D.val, pr=None)
pipe_feed_4.set_attr(D=pipe_feed_4.D.val, pr=None)
pipe_feed_5.set_attr(D=pipe_feed_5.D.val, pr=None)

pipe_return_0.set_attr(D=pipe_return_0.D.val, pr=None)
pipe_return_1.set_attr(D=pipe_return_1.D.val, pr=None)
pipe_return_2.set_attr(D=pipe_return_2.D.val, pr=None)
pipe_return_3.set_attr(D=pipe_return_3.D.val, pr=None)
pipe_return_4.set_attr(D=pipe_return_4.D.val, pr=None)
pipe_return_0.set_attr(D=pipe_return_5.D.val, pr=None)

pipe_feed_0.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)
pipe_feed_1.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)
pipe_feed_2.set_attr(
    Tamb=0,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)
pipe_feed_3.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)
pipe_feed_4.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_feed_5.set_attr(
    Tamb=0,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_feed_6.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_0.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_1.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_2.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_3.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_4.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_5.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

pipe_return_6.set_attr(
    Tamb=7,  # ambient temperature level in network's temperature unit
    kA="var"  # area independent heat transfer coefficient
)

nw.solve(mode="design")
nw.print_results()

pipe_feed_0.set_attr(Tamb=7, kA=pipe_feed_0.kA.val, Q=None)
pipe_feed_1.set_attr(Tamb=7, kA=pipe_feed_1.kA.val, Q=None)
pipe_feed_2.set_attr(Tamb=7, kA=pipe_feed_2.kA.val, Q=None)
pipe_feed_3.set_attr(Tamb=7, kA=pipe_feed_3.kA.val, Q=None)
pipe_feed_4.set_attr(Tamb=7, kA=pipe_feed_4.kA.val, Q=None)
pipe_feed_5.set_attr(Tamb=7, kA=pipe_feed_5.kA.val, Q=None)
pipe_feed_6.set_attr(Tamb=7, kA=pipe_feed_6.kA.val, Q=None)

pipe_return_0.set_attr(Tamb=7, kA=pipe_return_0.kA.val, Q=None)
pipe_return_1.set_attr(Tamb=7, kA=pipe_return_1.kA.val, Q=None)
pipe_return_2.set_attr(Tamb=7, kA=pipe_return_2.kA.val, Q=None)
pipe_return_3.set_attr(Tamb=7, kA=pipe_return_3.kA.val, Q=None)
pipe_return_4.set_attr(Tamb=7, kA=pipe_return_4.kA.val, Q=None)
pipe_return_5.set_attr(Tamb=7, kA=pipe_return_5.kA.val, Q=None)
pipe_return_6.set_attr(Tamb=7, kA=pipe_return_6.kA.val, Q=None)

nw.solve(mode="design")
nw.print_results()