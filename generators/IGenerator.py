import sys
sys.path.append("../scop_classification_by_PRoBERTa")
import pandas as pd
import os
import traceback

class IGenerator(object):
    def __init__(self) -> None:
        self.bad_file_path = "data/splits/bad_data.txt"
        self.tracebacks_file_path = "data/splits/tracebacks.txt"

    def do(self, pdb_id, chain_id, region):
        raise NotImplementedError()

    def do_linear(self, df, n_rows_to_skip, n_rows_to_evalutate, out_file_path=None):
        for i, row in df.iterrows():
            if i+1 <= n_rows_to_skip: continue
            
            self.do_distributed(i, df, out_file_path)

            print()
            if i+1 >= n_rows_to_skip+n_rows_to_evalutate: break
    
    def do_distributed(self, i, df, out_file_path=None):
        good_df, bad_df = pd.DataFrame(), pd.DataFrame()
        if out_file_path!=None and os.path.exists(out_file_path): 
            good_df = pd.read_csv(out_file_path)
        if os.path.exists(self.bad_file_path): 
            bad_df = pd.read_csv(self.bad_file_path)
            
        row = df.loc[i]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"]
        if chain_and_region.find(",")!=-1: return
        chain_and_region = chain_and_region.split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]
        if len(chain_id)>1: return
        
        # these pdbs does not exists
        if pdb_id=="6qwj": return
        if pdb_id=="1ejg": return
        if pdb_id=="7v7y": return
        if pdb_id=="3msz": return
        if pdb_id=="6l7f": return
        
        print(f"Row:{i+1} -> {pdb_id}:{chain_id}")
        try:
            self.do(pdb_id, chain_id, region)
        except Exception as e:
            if "DSSP failed" in str(e):
                bad_df = bad_df.append(df.loc[i], ignore_index=True)
                bad_df.reset_index(drop=True, inplace=True)
                bad_df.to_csv(self.bad_file_path, index=False)
            else:
                with open(self.tracebacks_file_path, "a") as f:
                    f.write(f"{i}, {pdb_id}, {chain_id}, {region}, {str(e)}")
                    f.write(traceback.format_exc())
            return
        
        if out_file_path!=None:
            good_df = good_df.append(df.loc[i], ignore_index=True)
            good_df.reset_index(drop=True, inplace=True)
            good_df.to_csv(out_file_path, index=False)

# gen = IGenerator()
# gen.do(4rek, A)