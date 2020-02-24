class Phases:
  def __init__(self, csvpath):
    import pandas as pd
    df = pd.read_csv(csvpath)

    self.tlsDict = {}

    #make a dict from dataframe
    for index, row in df.iterrows():
      tlsID = row['tlsID']
      phaseID = row['phaseID']
      if tlsID in self.tlsDict:
        self.tlsDict[tlsID].append(phaseID)
      else:
        self.tlsDict[tlsID]=[phaseID]

  def getTransition(self,tlsID,source,dest):
    #check arguments are correct
    if tlsID not in self.tlsDict:
      print("Error: tls %s does not exist" % str(tlsID))
      return False

    if source not in self.tlsDict[tlsID]:
      print("Error: source phase does not exist in tls %s" % str(tlsID))
      return False

    if dest not in self.tlsDict[tlsID]:
      print("Error: destination phase does not exist in tls %s" % str(tlsID))
      return False

    #create the transition name:
    transition = "TL_%s from %s to %s" % (str(tlsID),str(source),str(dest))
    return transition

  def getTlsPhases(self,tlsID):
    return self.tlsDict[tlsID]

  def getTls(self):
    return list(self.tlsDict.keys())
