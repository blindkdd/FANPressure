import set_sumo_home

import logging
import os
from time import sleep
import string
import random
import time

import xml.etree.ElementTree as ET

from shutil import copyfile

import numpy as np
import pandas as pd

import gym
from gym import spaces,error
from gym.utils import seeding

import traci
import traci.constants as tc
from tls import Phases

module_path = os.path.dirname(__file__)

class CornicheEnv(gym.Env):
    def __init__(self,oneway=True,uneven=False,GUI=True,minlength=1,macrostep=1,minsteps=1,predefined=False,scale=6):
        self.phases = Phases("./assets/tls_phases.csv")

        self.GUI = GUI
        self.predefined = predefined
        self.scale = scale

        self.steps_since_last_change = len(self.phases.getTls()) * [1]

        #used to force waiting in same phase if has not lasted enough, the action has no effect
        self.minlength = minlength

        #used to force waiting in same phase if has not lasted enough, the action is taken after
        #at least minsteps are past in the last phase
        self.minsteps = minsteps

        #used to make x micro steps in sumo, while it is only one step from the agent point of view
        self.macrostep=macrostep

        self._seed = 31337

        #how many SUMO seconds before next step (observation/action)
        self.SUMOSTEP = 1.0

        #In this version the observation space is the set of sensors
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,24), dtype=np.uint8)

        self.DETECTORS = []

        # DETDICT FORMAT EXAMPLE: self.DETDICT[(IntersectionID,DirectionID)]["in"] = ["e2Detector_gneE23_0_3","e2Detector_gneE24_0_22"]
        self.DETDICT = {}

        self.EDGESmain = []

        self.EDGES = []

        #Set action space as the set of possible phases
        self.action_space = [spaces.Discrete(len(self.phases.getTlsPhases(tl))) for tl in self.phases.getTls()]

        self.spec = {}

        #Generate an alphanumerical code for the run label (to run multiple simulations in parallel)
        self.runcode = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

        self.timestep = 0
        self.completed_trips = 0

        self._configure_environment()


    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _subscribeALL(self):
        for edge in self.allConnectedEdges:
            self.conn.edge.subscribe(edge, [tc.LAST_STEP_VEHICLE_NUMBER])

        for edge in self.junctionPreds:
            self.conn.edge.subscribe(edge, [tc.LAST_STEP_VEHICLE_NUMBER])

        for edge in self.allEdges:
            self.conn.edge.subscribe(edge, [tc.LAST_STEP_VEHICLE_NUMBER])

    def _configure_environment(self):
        if self.GUI:
            sumoBinary = set_sumo_home.sumoBinaryGUI
        else:
            sumoBinary = set_sumo_home.sumoBinary

        if self.predefined:
            self.cfgfile = "/assets/corniche_predefined.sumocfg"
        else:
            self.cfgfile = "/assets/corniche.sumocfg"

        self.argslist = [sumoBinary, "-c", module_path+self.cfgfile,
                             "--collision.action", "teleport",
            "--step-length", str(self.SUMOSTEP), "-S", "--time-to-teleport", "-1", "--scale", str(self.scale), #scale volume
            "--collision.mingap-factor", "0",
            "--collision.check-junctions", "true", "--no-step-log",
            "--vehroute-output", "vehicles.out.xml"]

        if self.GUI:
            self.argslist += ["--gui-settings-file", "viewsettings.xml"]

        traci.start(self.argslist,label=self.runcode)

        self.conn = traci.getConnection(self.runcode)

        time.sleep(5) # Wait for server to startup

        #get list of all edges
        self.EDGES = self.conn.edge.getIDList()

        #create dictionary for incoming and outgoing edges
        self.edgesDict = {}

        #get network file name
        cfg = ET.parse(module_path+self.cfgfile)
        config = cfg.getroot()
        name = config.find(".//input/net-file")
        name = name.get("value")

        #pressures observation
        tree = ET.parse(module_path+ "/assets/" + name)
        root = tree.getroot()

        #make a dataframe for lane counts
        allConnectedEdges = set()
        for x in root.findall('.//connection'):
            if x.attrib["from"][0]!=":":
                allConnectedEdges.add(x.attrib["from"])
            if x.attrib["to"][0]!=":":
                allConnectedEdges.add(x.attrib["to"])
        allConnectedEdges = sorted(allConnectedEdges)
        #allConnectedEdgesIndex = {x:i for i,x in enumerate(allConnectedEdges)}


        #create dictionary of edge predecessor internal lanes in junctions
        self.junctionPreds = set()
        self.junctionPair = {}
        for x in root.findall('.//connection[@via]'):
            if x.attrib["to"][0]!=":" and x.attrib["via"][0]==":" and x.attrib["via"][1:2]!="tl":
                pred = "_".join(x.attrib["via"].split("_")[:-1])
                edge = x.attrib["to"]
                self.junctionPreds.add(pred)
                self.junctionPair[edge] = pred
        self.junctionPreds = sorted(self.junctionPreds)



        allEdges = self.getAllNonInternalEdges(module_path+ "/assets/" + name)

        self.allConnectedEdges = allConnectedEdges
        self.allEdges = allEdges

        self._subscribeALL()

        if not self.predefined:

            for tlsID in self.phases.getTls():
                self.edgesDict[tlsID]={}
                #programs = root.findall(".//tlLogic[@id='%d']" % tlsID)
                #phases = set([int(program.get("programID").split(" ")[-1]) for program in programs])
                #phases = sorted(phases)
                pressures = []
                for phase in self.phases.getTlsPhases(tlsID):
                    self.edgesDict[tlsID][phase] = {"incoming":set(),"outgoing":set()}
                    state = root.findall(".//tlLogic[@programID='TL_%d from 0 to %d']/phase" % (tlsID,phase))[-1]
                    #state = state.find(".//phase[2]")
                    state = state.get("state")
                    leng = len(state)
                    for i in range(0, leng):
                        if state[i] == 'G':
                            data = root.find(".//connection[@tl='%d'][@linkIndex='%d']" % (tlsID, i))
                            connectionFROM = data.get('from')
                            connectionTO = data.get('to')
                            self.edgesDict[tlsID][phase]["incoming"].add(connectionFROM)
                            self.edgesDict[tlsID][phase]["outgoing"].add(connectionTO)

    def __del__(self):
        self.conn.close()

    def closeconn(self):
        self.conn.close()

    def _selectPhase(self,targets):
        for i,action in enumerate(targets):
            lastAction = int(self.conn.trafficlight.getProgram(str(i)).split(" ")[-1])
            if action != lastAction and self.steps_since_last_change[i]>= self.minlength:
                while self.steps_since_last_change[i] < self.minsteps:
                    #if min steps requirement is set, do other steps until it's satisfied
                    self.conn.simulationStep()
                    arrived = self.conn.simulation.getArrivedNumber()
                    self.completed_trips += arrived
                    self.steps_since_last_change = [x+1 for x in self.steps_since_last_change]
                    self.timestep +=1

                newprogram = "TL_%d from %d to %d" % (i,lastAction,action)
                self.conn.trafficlight.setProgram(str(i),newprogram)
                self.conn.trafficlight.setPhase(str(i),0)
                self.steps_since_last_change[i]=0
            elif self.conn.trafficlight.getPhase(str(i))==1:
                #extend the phase if it's not yellow (the length in tls definition may not be long
                #enough, thus it may skip to another phase by itself otherwise)
                self.conn.trafficlight.setPhaseDuration(str(i),60.0)
                #self.steps_since_last_change[i]+=1


    def _observeStateOLD(self):
        reward = []
        observations = []
        measures = {"completed_trips":self.completed_trips}

        if not self.predefined:
            for tlsID in self.phases.getTls():
                pressures = []
                for phase in self.phases.getTlsPhases(tlsID):
                    pressure = 0
                    for edge in self.edgesDict[tlsID][phase]["incoming"]:
                        pressure += self.conn.edge.getSubscriptionResults(edge)[tc.LAST_STEP_VEHICLE_NUMBER]
                    for edge in self.edgesDict[tlsID][phase]["outgoing"]:
                        pressure -= self.conn.edge.getSubscriptionResults(edge)[tc.LAST_STEP_VEHICLE_NUMBER]
                    pressures.append(pressure)
                observations.append(np.array(pressures))

        return np.array(observations),reward,measures

    def _observeState(self):
        reward = []
        observations = []
        measures = {"edgesIndex":self.allConnectedEdges}

        if not self.predefined:
            observations = [self.conn.edge.getSubscriptionResults(edge)[tc.LAST_STEP_VEHICLE_NUMBER] for edge in self.allConnectedEdges]
            observations_preds = [self.conn.edge.getSubscriptionResults(self.junctionPair[edge])[tc.LAST_STEP_VEHICLE_NUMBER] if edge in self.junctionPair.keys() else 0 for edge in self.allConnectedEdges]

            observations = np.array(observations_preds) + np.array(observations)

            #observations = [self.conn.lane.getSubscriptionResults(lane)[tc.LAST_STEP_OCCUPANCY] for lane in self.allConnectedLanes]


        return (observations,self._observeStateOLD()[0]),reward,measures


    def step(self, action):
        if not self.predefined:
            self._selectPhase(action)
        #self.conn.simulation.step(time=10.0)
        macrostep_arrived = 0
        macrostep_departed = 0
        for i in range(self.macrostep):
            self.conn.simulationStep()
            arrived = self.conn.simulation.getArrivedNumber()
            departed = self.conn.simulation.getDepartedNumber()

            macrostep_arrived += arrived
            macrostep_departed += departed
            self.completed_trips += arrived
            self.steps_since_last_change = [x+1 for x in self.steps_since_last_change]
            self.timestep +=1
        #get state and reward
        obs,reward,measures = self._observeState()
        measures['arrived'] = macrostep_arrived
        measures['departed'] = macrostep_departed
        episode_over = self.timestep >= (360000-1) or np.sum(reward) < -16
        if episode_over:
            self.conn.load(self.argslist[1:])
            self.timestep = 0
        return obs, reward, episode_over, measures

    def reset(self,restart=False):
        if restart:
            self.conn.load(self.argslist[1:])
            self._subscribeALL()
        self.timestep = 0
        return self._observeState()[0]

    def getAllNonInternalEdges(self,xmlpath):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xmlpath)
        net = tree.getroot()
        alledges = set()
        for x in net.findall(".//edge"):
            alledges.add(x.attrib["id"])
        internal = set()
        for x in net.findall('.//edge[@function="internal"]'):
            internal.add(x.attrib["id"])
        noninternal = list(alledges - internal)
        return noninternal
