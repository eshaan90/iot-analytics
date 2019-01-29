#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:02:14 2018

@author: MyReservoir
"""
import sys
import random
import math
random.seed(10)

def service_completion(queue,service, mc,dc,CLS):
    if queue==0:
        service=0
    else:
        service=1
        queue=0
        CLS=mc+dc
    return (queue,service,CLS)

def arrival(queue,service,mc,dc,rc,CLS,CLR):
    if service==0: 
        service=1
        CLS=mc+dc
    else:
        if queue==1: 
            CLR.append(int(mc+rc))
        else:       
            queue=1
    return (queue,service,CLS,CLR)

def retransmission(queue,service,B,mc,dc,rc,CLS,CLR):
    if queue+service<B: 
        if service==0: 
            service=1
            CLS=mc+dc
        else: 
            queue=1
    else:  
        CLR.append(mc+rc)
    return(queue,service,CLS,CLR)
    

def print_output(mc,CLA,CLS,service,queue,CLR):
    f=open('output.txt','a')
    
    if  CLR==[]:
        f.write("\n" +"mc= {}, CLA= {}, CLS= {}, No. of requests= {}, CLR= {}".format(mc,CLA,CLS,service+queue,'-'))
    else:   
        f.write("\n"+"mc= {}, CLA= {}, CLS= {}, No. of requests= {}, CLR= {}".format(mc,CLA,CLS,service+queue,CLR))
    f.close()

def find_min(CLA,CLS,CLR):
    if CLR==[]:
        return min(CLA,CLS)
    else:
        return min(CLA,CLS,CLR[0])
    

def parse_input():
    ac=int(sys.argv[1])
    rc=int(sys.argv[2])
    dc=int(sys.argv[3])
    B=int(sys.argv[4])
    clock_end=int(sys.argv[5])
    return (ac,rc,dc,B,clock_end)
    
def main():
    mc=0
    CLR=[]
    CLS=100000
    service=0
    queue=0
    (ac,rc,dc,B,clock_end)=parse_input()
    
    r=random.uniform(0,1)
    ac=int(-1*ac*math.log(r))
    rc=int(-1*rc*math.log(r))
    
    while mc<=clock_end:
        if mc==0:
            CLA=2
            f=open('output.txt','w')
            f.write("\n" +"mc= {}, CLA= {}, CLS= {}, No. of requests= {}, CLR= {}".format(mc,CLA,'-',service+queue,'-'))
            mc=find_min(CLA,CLS,CLR)
            f.close()
            
        if CLR!=[]:           
            if CLR[0]<=CLA and CLR[0]<=CLS:
                mc=CLR.pop(0)
                (queue,service,CLS,CLR)=retransmission(queue,service,B,mc,dc,rc,CLS,CLR)
                print_output(mc,CLA,CLS,service,queue,CLR)
                mc=find_min(CLA,CLS,CLR)
                
            elif CLA<CLR[0] and CLA<=CLS:
                if CLR[0]==1000 and service+queue==2:
                    CLR.pop(0)
                (queue,service,CLS,CLR)=arrival(queue,service,mc,dc,rc,CLS,CLR)
                CLA=mc+ac
                print_output(mc,CLA,CLS,service,queue,CLR)
                mc=find_min(CLA,CLS,CLR)
                     
            else:
                (queue,service,CLS)=service_completion(queue, service, mc, dc, CLS)
                print_output(mc,CLA,CLS,service,queue,CLR)
                mc=find_min(CLA,CLS,CLR)
        
        else:
            if CLA<=CLS:
                (queue,service,CLS,CLR)=arrival(queue,service,mc,dc,rc,CLS,CLR)
                CLA=mc+ac
                print_output(mc,CLA,CLS,service,queue,CLR)
                mc=find_min(CLA,CLS,CLR)
            else:
                (queue,service,CLS)=service_completion(queue, service, mc, dc, CLS)
                print_output(mc,CLA,CLS,service,queue,CLR)
                mc=find_min(CLA,CLS,CLR)

 
if __name__ == '__main__':
    main()