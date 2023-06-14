import pandas as pd
import numpy as np
import os




def write_csv(path,data):
    
    df = pd.DataFrame(data)
    if (os.path.exists(path)):
        df.to_csv(path,index=False)
    else:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        print(directory)
        df.to_csv(path,index=False)



# For testing

if __name__ == "__main__":
    
    data = [["123", "456", "789"]]
    col_names = ["Power", "Energy", "Latency"]
    my_actions = {'PagePolicy': 'Open', 'Scheduler': 'FrFcfsGrp', 'SchedulerBuffer': 'Shared', 'RequestBufferSize': 8, 'CmdMux': 'Oldest', 'RespQueue': 'Fifo', 'RefreshPolicy': 'NoRefresh', 'RefreshMaxPostponed': 7, 'RefreshMaxPulledin': 8, 'PowerDownPolicy': 'NoPowerDown', 'Arbiter': 'Simple', 'MaxActiveTransactions': 16}
    my_obs = {'Energy': 250557975.0, 'Power': 2647.66, 'Latency': 94633750.0}

    data = {**my_actions, **my_obs}
    df = pd.DataFrame([data])
    print(df)
    

    
    


