import os
import zipfile
import scipy.io as sio
import io

base_path = "."

def check_deep_folders():
    print("--- Explorando as entranhas da Ninapro (Zips) ---")
    databases = sorted([f for f in os.listdir(base_path) if "Ninapro_DB" in f])
    
    for db in databases:
        db_path = os.path.join(base_path, db)
        zips = [f for f in os.listdir(db_path) if f.endswith('.zip')]
        
        print(f"\n{db}: Encontrados {len(zips)} arquivos .zip")
        
        if zips:
            # Vamos espiar apenas o primeiro zip de cada DB para ver o que tem dentro
            primeiro_zip = os.path.join(db_path, zips[0])
            try:
                with zipfile.ZipFile(primeiro_zip, 'r') as z:
                    # Lista arquivos .mat dentro do zip
                    conteudo_mat = [f for f in z.namelist() if f.endswith('.mat')]
                    print(f"  Exemplo [{zips[0]}]: contém {len(conteudo_mat)} arquivos .mat")
                    
                    if conteudo_mat:
                        # TESTE DE LEITURA: Lendo o .mat direto da memória (sem extrair pro disco)
                        with z.open(conteudo_mat[0]) as mat_file:
                            # Precisamos do io.BytesIO porque o scipy espera um "file-like object"
                            data = sio.loadmat(io.BytesIO(mat_file.read()))
                            # Geralmente Ninapro tem chaves como 'emg', 'stimulus', 'repetition'
                            vars_sinal = [k for k in data.keys() if not k.startswith('__')]
                            print(f"  Variáveis encontradas no sinal: {vars_sinal}")
            except Exception as e:
                print(f"  Erro ao ler {zips[0]}: {e}")

if __name__ == "__main__":
    check_deep_folders()