import numpy as np;
import scipy as sp;
import matplotlib.pyplot as plt
import sys

from scipy import signal;

if len(sys.argv)<3:
    print('Usage: "DDCFirGen.py [decimation factor] [cutoff frequency]"')
else: 
    cutoff = float(sys.argv[2]);
    M = float(sys.argv[1]);
    
    fs = 65;

    if M%1 == 0.0:
        n = 16*M
    elif M%1 == 0.5:
        n = 32*M
    elif M%1 == 0.25:
        n = 64*M
    elif M%1 == 0.75:
        n = 64*M
        
    n = int(n);
    
    h = signal.firwin(n, cutoff/(fs/2));
    Mh = max(h);
    sf = 32624/Mh;
    h2 = sf*h;
    h3 = h2[int(n/2):];
    
    out = open("fir_generated.txt", "w")
    for n in range (0, int(n/2)):
        out.write(str(int(h3[n])));
        out.write('\n');
 



