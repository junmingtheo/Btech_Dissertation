import numpy as np 
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
from pathlib import Path

def process_chunk(chunk_data):
    simulator = datagen()  
    return [simulator.simulate_row(row) for row in chunk_data]

class datagen:
    num_processes = mp.cpu_count()
    T_0 = 300.0
    V = 1.0
    k_0 = 8.46*(np.power(10,6))
    C_p = 0.231
    rho_L = 1000.0
    Q_s = 0.0
    T_s = 402.0
    F = 5.0
    E = 5*(np.power(10,4))
    delta_H = -1.15*(np.power(10,4))
    R = 8.314
    C_A0s = 4.0
    C_As = 1.95
    t_final = 0.01
    t_step = 1e-4
    n = 100.0 # int(t_final/t_step)
    P = np.array([[1060.0, 22.0], [22.0, 0.52]])
    u1_df= pl.DataFrame({"C_A0":np.linspace(-3.5, 3.5, 30, endpoint=True)})
    u2_df = pl.DataFrame({"Q":np.linspace(-5e5, 5e5, 30, endpoint=True)})
    T_initial_df = pl.DataFrame({"T_initial":(np.linspace(300, 600, 50, endpoint=True) - T_s).reshape(-1)})
    CA_initial_df = pl.DataFrame({"CA_initial":(np.linspace(0, 6, 50, endpoint=True) - C_As).reshape(-1)})
    df = u1_df.join(u2_df.join(T_initial_df.join(CA_initial_df, how="cross"),how="cross"),how="cross")
    df = df.with_columns((P[0][0]*pl.col("CA_initial")**2 + P[0][1]*pl.col('CA_initial')*pl.col('T_initial') + P[1][0]*pl.col('CA_initial')*pl.col('T_initial') + P[1][1]*pl.col("T_initial")**2).alias("V(x)"))
    df = df.filter(pl.col("V(x)")<372.0)

    def __init__(self):
        pass


    def simulate_row(self,row):
        return self.__simulation(row["CA_initial"],row['T_initial'],row['C_A0'],row['Q'])



    """can parallelize this"""
    def __simulation(self,C_A_init,T_init,C_A0,Q):
        # C_A_init=C_A_init[0]
        # T_init=T_init[0]
        # C_A0=C_A0[0]
        # Q=Q[0]
        T_list = []
        C_A_list=[]
        C_A = C_A_init + self.C_As
        T = T_init + self.T_s
        
  
        for i in range(int(self.t_final/self.t_step)):

            dCAdt = self.F / self.V * (C_A0 - C_A) - self.k_0 * np.exp(-self.E / (self.R * T)) * C_A**2
            dTdt = self.F / self.V * (self.T_0 - T) - self.delta_H / (self.rho_L * self.C_p) * self.k_0 * np.exp(-self.E / (self.R * T)) * C_A**2 + Q / (self.rho_L * self.C_p * self.V)
            C_A += dCAdt * self.t_step
            T += dTdt * self.t_step    
            T_list.append(T)
            C_A_list.append(C_A)
            # if (i+1)% 10 == 0:
            #      T_list.append(T-self.T_s)
            #      C_A_list.append(C_A -self.C_As)

            
        T_list = np.array(T_list) - self.T_s
        C_A_list = np.array(C_A_list) - self.C_As
        T_list=np.array(T_list)[::10].tolist()
        C_A_list=np.array(C_A_list)[::10].tolist() 
             
                       
        return [C_A_list,T_list]
    


    def parallel_map_elements(self, df):

        chunk_size = max(1, len(df) // self.num_processes)
        chunks = [df.slice(i, chunk_size).to_dicts() for i in range(0, len(df), chunk_size)]

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:

            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
                
            

        return results

    # def testest(self):
    #     new_DF = []
    #     for i in range(self.df.shape[0]):
    #         old_df = self.df[i]
    #         new_DF.append(self.simulate_row(old_df))
    #     new_DF = pl.DataFrame({'simulation':new_DF})

    #     return new_DF

    def apply_simulation(self):
        
        self.df= self.df.with_columns(pl.col("Q")+self.Q_s)
        self.df= self.df.with_columns(pl.col("C_A0")+self.C_A0s)

        results = self.parallel_map_elements(self.df)
        result_df = self.df.with_columns(
            pl.Series(name='simulation_result', values=results, dtype=pl.List)) 
        

        self.df = result_df.with_columns([
            pl.col('simulation_result').list.get(0).alias('CA_N_timestep'),
            pl.col('simulation_result').list.get(1).alias('T_N_timestep')
        ]).drop('simulation_result')

        self.df= self.df.with_columns(pl.col("Q")-self.Q_s)
        self.df= self.df.with_columns(pl.col("C_A0")-self.C_A0s)

        
    
    def retrieve_df(self):
        return self.df
    
    def df_to_tensor(self):
        self.nn_input  = np.repeat(self.df.select(["CA_initial","T_initial","C_A0","Q"]).to_numpy().reshape(-1,1,4),10,axis=1)
        self.nn_output =  self.df.select(['CA_N_timestep', 'T_N_timestep']).explode(pl.all()).to_numpy().reshape(-1,10,2)
    

    def standard_normalization(self):
        try:
            self.nn_input

        except AttributeError:

            print("run df_to_tensor first")

        else: 
            
            try :
                 self.X_train 
                 
            except AttributeError:

                """did split first to prevent data leakage"""    
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.nn_input, self.nn_output, test_size=0.2,random_state=123) 

                x_scaler = StandardScaler().fit(self.X_train.reshape(-1,4))
                y_scaler = StandardScaler().fit(self.y_train.reshape(-1,2))

                self.X_train_norm = x_scaler.transform(self.X_train.reshape(-1,4)).reshape(-1,10,4)  
                self.y_train_norm = y_scaler.transform(self.y_train.reshape(-1,2)).reshape(-1,10,2) 
                self.X_test_norm = x_scaler.transform(self.X_test.reshape(-1,4)).reshape(-1,10,4)  
                self.y_test_norm = y_scaler.transform(self.y_test.reshape(-1,2)).reshape(-1,10,2) 
                # self.X_scaler_mean = x_scaler.mean_
                # self.y_scaler_mean = y_scaler.mean_
                # self.X_scaler_std = np.sqrt(x_scaler.var_)
                # self.y_scaler_std = np.sqrt(y_scaler.var_)
                
                Path("standardscaler").mkdir(exist_ok=True)
                # np.save('standardscaler/xscalermean.pkl',self.X_scaler_mean)
                # np.save('standardscaler/yscalermean.pkl',self.y_scaler_mean)
                # np.save('standardscaler/xscalerstd.pkl',self.X_scaler_std)
                # np.save('standardscaler/yscalerstd.pkl',self.y_scaler_std)
                joblib.dump(x_scaler, 'standardscaler/xscaler.pkl')
                joblib.dump(y_scaler, 'standardscaler/yscaler.pkl')

            else:     
                print("no need to run again")
                return

                
            
