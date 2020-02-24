import os, sys
if 'SUMO_HOME' not in os.environ:
     os.environ["SUMO_HOME"]="/Users/srizzo/sumo"
     #sys.path.append('SUMO_HOME')

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
sumoBinary = "/usr/local/bin/sumo"
sumoBinaryGUI = "/usr/local/bin/sumo-gui"
