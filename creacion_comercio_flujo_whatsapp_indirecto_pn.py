# Databricks notebook source
# MAGIC %pip install nltk

# COMMAND ----------

!pip install --upgrade google-api-python-client

# COMMAND ----------

# DBTITLE 1,Instalar Paquetes
!pip install googlemaps

# COMMAND ----------

!pip install python-Levenshtein

# COMMAND ----------

# DBTITLE 1,Reinicio Libreria
dbutils.library.restartPython()

# COMMAND ----------

import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'slack_sdk'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])

# COMMAND ----------

# DBTITLE 1,Load SIGNIO class
import requests
from requests.auth import HTTPBasicAuth

# Configuración de Bitbucket
username = 'data_puntored'
app_password = (dbutils
                  .secrets
                  .get(scope="data_secrets",
                  key="APP_PASSWORD_BITBUCKET")
                  )
repo_owner = 'brainwinner'
repo_slug = 'data'
branch_name = 'master'
file_path = 'motor_credito/signio_api_veci_funciones.py' #'motor_credito/truora_api.py'
local_file_name = '/tmp/signio_api_veci_funciones.py' #'/tmp/truora_api.py'

# URL de la API de Bitbucket para descargar el archivo
url = f"https://api.bitbucket.org/2.0/repositories/{repo_owner}/{repo_slug}/src/{branch_name}/{file_path}"

# Realizar la solicitud para descargar el archivo
response = requests.get(url, auth=HTTPBasicAuth(username, app_password))

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    with open(local_file_name, 'wb') as f:
        f.write(response.content)
    print(f"El archivo '{file_path}' ha sido descargado como '{local_file_name}'")
    import sys
    sys.path.insert(0, '/tmp')
    # from truora_api import *
else:
    raise EOFError(f"Error al descargar el archivo: {response.status_code}")


# COMMAND ----------

from signio_api_veci_funciones import *

# COMMAND ----------

# DBTITLE 1,client y librerias
import boto3
import requests
import json
from datetime import datetime, timedelta
import re
import pandas as pd
# from datetime import datetime
import Levenshtein
from time import sleep
import googlemaps

api_key = dbutils.secrets.get(scope="data_secrets", key="API_KEY_GOOGLE_MAPS")  # Reemplaza esto con tu clave de API de Google Maps
gmaps = googlemaps.Client(key=api_key)



AWS_ACCESS_KEY = (dbutils
                  .secrets
                  .get(scope="data_secrets",
                  key="AWS_ACCESS_KEY")
                  )
AWS_SECRET_KEY = (dbutils
                  .secrets
                  .get(scope="data_secrets",
                  key="AWS_SECRET_KEY")
                  )

session = boto3.session.Session()
client = session.client('s3',
                        region_name='us-east-1',
                        endpoint_url='https://s3.amazonaws.com',
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY
                        )

# COMMAND ----------

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_stopwords(text):
    stop_words = set(stopwords.words('spanish'))
    filtered_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return filtered_text

# COMMAND ----------

df_mun = spark.sql("""
                   SELECT 
                   c.name AS mun
                   ,c.code AS mun_code
                   ,d.name AS depar
                   ,d.code AS depar_code
                    FROM
                        bronze.veci.dane_cities AS c
                    LEFT JOIN 
                        bronze.veci.dane_departments AS d
                        ON c.dane_department_id = d.id
                """).toPandas().replace("N. DE SANTANDER","NORTE DE SANTANDER").replace("S.","SUR")

df_mun['depar'] = df_mun['depar'].apply(lambda x: remove_stopwords(x))
df_mun['mun'] = df_mun['mun'].apply(lambda x: remove_stopwords(x))


df_mun.set_index(["depar","mun"], inplace=True)


# COMMAND ----------

df_estable = spark.sql("""
SELECT
  TIES_ID
  ,TIES_DESCRIPCION
  FROM
    bronze.cxr_web.tipo_establecimiento
""").toPandas().set_index("TIES_DESCRIPCION")


# COMMAND ----------

import unicodedata
import re

def clean_text(text):
    # Normalize the text to decompose special characters
    text = unicodedata.normalize('NFD', text)
    # Remove accents and special characters
    text = re.sub(r'[\u0300-\u036f]', '', text)
    # Remove any remaining non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# COMMAND ----------

def calcular_digito_verificacion(nit):
    # Convertir el NIT a una cadena para manejar la longitud
    nit_str = str(nit)
    
    # Multiplicadores estándar
    multiplicadores = [3, 7, 13, 17, 19, 23, 29, 37, 41, 43]
    
    # Si el NIT tiene más de 10 dígitos, repetimos los multiplicadores desde el inicio
    multiplicadores_usados = multiplicadores * ((len(nit_str) + len(multiplicadores) - 1) // len(multiplicadores))
    
    # Ajustamos la longitud de los multiplicadores para que coincida exactamente con la del NIT
    multiplicadores_usados = multiplicadores_usados[-len(nit_str):]
    
    # Invertir los dígitos del NIT
    nit_reverso = nit_str[::-1]
    
    # Calcular la suma de los productos del NIT por los multiplicadores
    suma_productos = sum(int(digito) * multiplicador for digito, multiplicador in zip(nit_reverso, multiplicadores_usados))
    
    # Calcular el dígito de verificación
    residuo = suma_productos % 11
    digito_verificacion = 11 - residuo if residuo > 1 else 0
    
    return digito_verificacion

# Ejemplo de uso
# nit = 1055918346
# digito_verificacion = calcular_digito_verificacion(nit)
# print(f"El dígito de verificación para el NIT {nit} es: {digito_verificacion}")


# COMMAND ----------

# DBTITLE 1,Muestra_peticion
# {
#     “idDistribuidor”: “322978",
#     “tipoPersona”: “2",
#     “tipoDocumento”: “1",
#     “numeroDocumento”: “860500600",
#     “digitoVerificacion”: “5",
#     “nombresPersona”: “Representante”,
#     “apellidosPersona”: “Legal”,
#     “fechaNacimiento”: “1995-08-25",
#     “celular”: “3115368690",
#     “direccion”: “cll 100 # 100 - 100",
#     “email”: “donaton@dontaon.co”,
#     “idtipoEstablecimiento”: “8",
#     “nombreEstablecimiento”: “DONATON S.A.S”,
#     “iddepto”: “11",
#     “idCiudad”: “11001",
#     “latitud”: “7.137264320",
#     “longitud”: “-73.123278",
#     “codigoCiiu”: “2930",
#     “datosRegimen”: “1",
#     “regimenIVA”: “4",
#     “declarante”: “0",
#     “autorretenedor”: “1",
#     “grancontribuyente”: “1",
#     “ley1429”: “1",
#     “regimenespecial”: “1",
#     “origenCreacion”: “AUTOAFILIACION_TRUORA”
# }

# COMMAND ----------

# DBTITLE 1,Función envio comunicaciones
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def send_message_slack(channel = 'flujo_vinculacion_whatsapp',
                       message = None,
                       title = None,
                       filepath = None,
                       ):
    
    if message is None:
        message = f"""
            No mensaje 
        """
    
    # iris.to_excel(filepath,index=False)

    TOKEN_USER = dbutils.secrets.get(scope="data_secrets", key="TOKEN_BOT_REPORTES_SLACK")
    client = WebClient(token=TOKEN_USER)
    try:
        response_msg = client.chat_postMessage(channel=channel,
                                            text = message)
        response = client.files_upload(channels=channel,title=title, file=filepath)
        assert response["file"]  # the uploaded file
        # print(f"fecha: {liqui_month}  \t   Colaborador: {colaborador}  \t  canal:{channel} \t file:{filepath}")
    except SlackApiError as e:
        assert e.response["ok"] is False
        assert e.response["error"]
        print(f"Got an error: {e.response['error']}")

# COMMAND ----------

# DBTITLE 1,Activar Infra para creación de comercios
import requests

def activate_creation_comercie(api_key):
    header = {"X-Api-Key": api_key}
    response = requests.get(url="https://onboarding-api.puntored.co/api/v1/ping/", headers=header)
    if response.status_code == 200:
        return True,response
    else:
        return False,response

TOKEN_API_KEY = dbutils.secrets.get(
    scope="data_secrets", key="API_KEY_CREACION_COMERCIOS"
)

# COMMAND ----------

# DBTITLE 1,Funciones
TOKEN_API_KEY = dbutils.secrets.get(
    scope="data_secrets", key="API_KEY_CREACION_COMERCIOS"
)
TRUORA_API_KEY = dbutils.secrets.get(scope="data_secrets", key="APP_PASSWORD_TRUORA")

SIGNIO_EMAIL =  dbutils.secrets.get(scope="data_secrets", key="EMAIL_SIGNIO_VINCULACION_TAT_DIRECTO")
SIGNIO_PASSWORD = dbutils.secrets.get(scope="data_secrets", key="PASSWORD_SIGNIO_VINCULACION_TAT_DIRECTO") 

class create_comercie:
    def __init__(self,
                 token_api_key,
                 truora_api_key,
                 client,
                 signio_email,
                 signio_password):
        self.token_api_key = token_api_key
        self.truora_api_key = truora_api_key
        self.client = client
        self.signio = signio_api(signio_email = signio_email,
                                signio_password = signio_password,
                                tenant = "PUNTORED - OPERACIONES",
                                )

    def load_logs(
        self,
        file,
        bucket_name,
        ):

        path = f"/tmp/{file.split('/')[-1].split('.')[0]}_v1.json"
        found = False
        try:
            client.download_file(bucket_name, f"{file}", path)
            found = True
        except Exception as e:
            print("Exception caught: ", e)
            pass
        if found:
            with open(path, "r") as f:
                js = json.load(f)
        else:
            js = None
        return js

    def get_validation(self,idp):
        ENDPOINT = f"https://api.identity.truora.com/v1/processes/{idp}/result"
        headers = {"Truora-API-Key": self.truora_api_key}
        response = requests.get(ENDPOINT, headers=headers)
        if response.status_code == 200:
            mdict = response.json()
            name = mdict['first_name']
            lastname = mdict['last_name']
        else:
            name = "NOT FOUND"
            lastname = "NOT FOUND"
        return name,lastname

    def get_check_id(self,idp):
        ENDPOINT = f"https://api.identity.truora.com/v1/processes/{idp}/result"
        headers = {"Truora-API-Key": self.truora_api_key}
        response = requests.get(ENDPOINT, headers=headers)
        if response.status_code == 200:
            mdict = response.json()
            try:
                check_id = mdict['validations'][0]['details']['background_check']['check_id']
            except:
                check_id = None
        else:
            check_id = None
        return check_id
    
    def get_check_background_results(self,check_id):
        ENDPOINT = f"https://api.checks.truora.com/v1/checks/{check_id}"
        headers = {"Truora-API-Key": self.truora_api_key}
        response = requests.get(ENDPOINT, headers=headers)
        if response.status_code == 200:
            mdict = response.json()
            try:
                score_global = mdict["check"]["score"]
            except:
                score_global = None
            try:
                score_criminal_records = mdict["check"]["scores"][1]["score"]
                severity_criminal_records = mdict["check"]["scores"][1]["severity"]
            except:
                score_criminal_records = None
                severity_criminal_records = None
            try:
                score_legal_background = mdict["check"]["scores"][2]["score"]
                severity_legal_background = mdict["check"]["scores"][2]["severity"]
            except:
                score_legal_background = None
                severity_legal_background = None
        else:
            
            score_global  = None
            score_criminal_records  = None
            severity_criminal_records  = None
            score_legal_background  = None
            severity_legal_background  = None
        
        result = {"score_global": score_global,
                "score_criminal_records": score_criminal_records,
                "severity_criminal_records": severity_criminal_records,
                "score_legal_background": score_legal_background,
                "severity_legal_background": severity_legal_background,}
        return result


    def get_departamento_ciudad(self,latitud,longitud):
        try:
            departamento, ciudad,ciudad2 = None,None,None
            reverse_geocode_result = gmaps.reverse_geocode((latitud, longitud))
            print(reverse_geocode_result[0]['address_components'])
            for component in reverse_geocode_result[0]['address_components']:
                if 'locality' in component['types']:
                    ciudad = component['long_name'].upper()
                if 'administrative_area_level_2' in component['types']:
                    ciudad2 = component['long_name'].upper()
                if 'administrative_area_level_1' in component['types']:
                    departamento = component['long_name'].upper()
                # if 'country' in component['types']:
                #     country = component['long_name']
            
            if ciudad is None and ciudad2 is not None:
                ciudad = ciudad2
        except Exception as e:
            print(f"Sin respuesta: {e}")
            departamento = None
            ciudad = None
        return departamento, ciudad, ciudad2,


    def create_comercie(self, js,df_mun):
        """
        RETURN:
        (cod_int,dict): cod_int 1, creacion ok, cod_int 0 error creacion vía API, cod_int -1, sin firma de documentos
        """
        id_distribuidor = js['id_promotor']
        js_aux = js.copy()
        idp_flujo1 = js['id_proceso_flujo1']
        name, lastname = self.get_validation(idp_flujo1)
        cedula = js['cedula']
        tipo_documento = js['tipo_documento']
        telefono = js['telefono']
        fecha_nacimiento = js["fecha_nacimiento"]
        fecha_expedicion = js["fecha_expedicion"]
        genero = js["genero"]
        correo = js["correo"]
        actividad_economica = js["actividad_economica"]
        direccion_ubicacion = js["direccion_ubicacion"]
        departamento_ = js["departamento"].upper()
        ciudad_ = js["ciudad"].upper()
        barrio = js["barrio"]
        direccion = js["direccion"]
        estrato = js["estrato"]
        ppe = js["ppe"]
        tipo_establecimiento = js["tipo_establecimiento"]
        otro_establecimiento = js["otro_establecimiento"]
        tipo_establecimiento = tipo_establecimiento if tipo_establecimiento != "OTRO" else otro_establecimiento.upper()
        razon_social_aliado = js["razon_social_aliado"]
        id_documento_contrato = js["id_documento_contrato"]
        id_firma_electronica_usuario = js["id_firma_electronica_usuario"].replace(" ","")
        id_firma_promotor = js["id_firma_electronica_promotor"].replace(" ","")
        signed_by = js["signed_by"]
        fecha = js["fecha"]
        check_id = self.get_check_id(idp_flujo1)
        if check_id is not None:
            result_checks = self.get_check_background_results(check_id)
            score_global = result_checks["score_global"]
            have_score_global = score_global is not None
        else:
            have_score_global = False
        if have_score_global:
            cumple = float(score_global) >= 0.8
        else:
            cumple = False
        dv = str(calcular_digito_verificacion(cedula))
        if id_documento_contrato == "null":
            return -2,None 
        # id Establecimiento
        tipo_establecimiento = "OTRO" if  tipo_establecimiento not in df_estable.index else  tipo_establecimiento  
        if tipo_establecimiento != "OTRO":
            id_establecimiento = df_estable.loc[tipo_establecimiento]["TIES_ID"]
            if type(id_establecimiento) == pd.Series:
                id_establecimiento = id_establecimiento[0]
        else:
            id_establecimiento = df_estable.loc["OTRO"]["TIES_ID"]


        #Validacion firma
        num_firmas,docs = (self
                           .signio
                           .consultar_transaccion(id_transaccion=id_documento_contrato,
                                                  id_firmante=id_firma_electronica_usuario,
                                                  return_details=True
                                                  )
                            )
    
        num_firmas_p,docs_p = (self
                           .signio
                           .consultar_transaccion(id_transaccion=id_documento_contrato,
                                                  id_firmante=id_firma_promotor,
                                                  return_details=True
                                                  )
                            )
        
        df_docs = pd.DataFrame(docs).transpose().reset_index(names='nombre')
        df_docs['nombre'] = df_docs['nombre'].astype(str)  # Convert the column to string
        print(df_docs)
        df_docs_summary = (df_docs
                        .groupby(["firmado"])
                        .agg(count=('nombre', 'count'),
                            docs=('nombre', lambda x: " | ".join(x.tolist()))
                            )
                        )
        df_docs_p = pd.DataFrame(docs_p).transpose().reset_index(names='nombre')
        df_docs_p['nombre'] = df_docs_p['nombre'].astype(str)  # Convert the column to string
        print(df_docs_p)
        df_docs_summary_p = (df_docs_p
                        .groupby(["firmado"])
                        .agg(count=('nombre', 'count'),
                            docs=('nombre', lambda x: " | ".join(x.tolist()))
                            )
                        )
        df_docs_summary['rol'] = "PROMOTOR"
        df_docs_summary_p['rol'] = "PARTICIPE"
        df_docs_summary = df_docs_summary.append(df_docs_summary_p)
        firmado = num_firmas==4
        if not(firmado):
            docs_pendientes = df_docs_summary.loc[False] # Documentos pendientes
            docs_pendientes['cedula'] = cedula 
            docs_pendientes['telefono'] = telefono 
        else:
            docs_pendientes = None
        # extract lat, lon
        pattern = r'query=([-+]?\d*\.\d+|\d+),([-+]?\d*\.\d+|\d+)'

        # Buscar coincidencias en la URL
        match = re.search(pattern, direccion_ubicacion)
        # print(match)
        if match:
            latitude = str(match.group(1))
            longitude = str(match.group(2))
            departamento,ciudad,ciudad2 = self.get_departamento_ciudad(latitude, longitude)

            departamento = departamento if departamento  else None
            ciudad = ciudad if ciudad  else None
            
        else:
            latitude = "0.00"
            longitude = "0.00"
            departamento,ciudad = "BOGOTA","BOGOTA, D.C."
        # print("antes de procesar", departamento,ciudad,ciudad2)
        if departamento is not None and ciudad is not  None:
            # print(departamento,ciudad)
            departamento = departamento.upper()
            ciudad = ciudad.upper()
            # ciudad = ciudad2.upper()
            departamento = departamento.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
            ciudad = ciudad.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
            # ciudad2 = ciudad2.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")

            ciudad  = ciudad if ciudad != "BOGOTA" else "BOGOTA, D.C."
        else:
            # print(departamento,ciudad)
            departamento = departamento_.upper()
            ciudad = ciudad_.upper()
            # ciudad = ciudad2.upper()
            departamento = departamento.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
            ciudad = ciudad.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
            # ciudad2 = ciudad2.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")

            ciudad  = ciudad if ciudad != "BOGOTA" else "BOGOTA, D.C."
        #Extraer codigos de mun y depart
        # print(departamento,ciudad)
        found_cods = True
        departamento = remove_stopwords(departamento)
        ciudad = remove_stopwords(ciudad)
        ciudad2 = remove_stopwords(ciudad)
        # print("listo: ",departamento,ciudad)
        cumple = (departamento,ciudad) in df_mun.index
        print("cumple deparatmento_ciudad: ",cumple)
        if cumple:
            
            mun_code, depar_code = df_mun.loc[(departamento,ciudad)][["mun_code", "depar_code"]]
            print(mun_code, depar_code)
        else:
            mun_code, depar_code = "001","11"
            found_cods = False
        
        ENDPOINT = "https://commerce-create-api-prod-11774552561.us-east1.run.app/api/v1/commerce/commerce/"
        headers = {"X-Api-Key": self.token_api_key}
        body = {
            "idDistribuidor": f"{id_distribuidor}",
            "tipoPersona": "1",
            "tipoDocumento": "2" if tipo_documento in ["national-id","CC"] else "3",
            "numeroDocumento": cedula,
            "digitoVerificacion": dv,
            "nombresPersona": clean_text(name),
            "apellidosPersona": clean_text(lastname),
            "fechaNacimiento": fecha_nacimiento.split('T')[0],
            "celular": telefono,
            "direccion": " ".join([word for word in clean_text(direccion).split(" ") if word != ""]),
            "barrio": " ".join([word for word in clean_text(barrio).split(" ") if word != ""]),
            "email": correo,
            "idtipoEstablecimiento": f"{id_establecimiento}",
            "nombreEstablecimiento": clean_text(razon_social_aliado)[0:119] if len(clean_text(razon_social_aliado))>119 else clean_text(razon_social_aliado),
            "iddepto": depar_code,
            "idCiudad": f"{depar_code}{mun_code}",
            "latitud": latitude,
            "longitud": longitude,
            "codigoCiiu": clean_text(actividad_economica),
            "datosRegimen": "0",
            "regimenIVA": "1",
            "declarante": "0",
            "autorretenedor": "0",
            "grancontribuyente": "0",
            "ley1429": "0",
            "regimenespecial": "0",
            "origenCreacion": "AUTOAFILIACION_WHATSAPPs_TRUORA",
        }
        print(f"body enviado:{body}")
        if firmado and cumple:
            response = requests.post(ENDPOINT, json=body, headers=headers)
            if response.status_code == 200:
                js_aux.update({"founded_cod":found_cods,
                                    "respuesta_api":response.text,
                                    'success':True})
                return 1,js_aux
            else:
                js_aux.update({"founded_cod":found_cods,
                                    "respuesta_api":response.text,
                                    "success":False}
                                   )
                return 0,js_aux
        elif not(cumple) and have_score_global:
            js_aux.update({"founded_cod":found_cods,
                                    "respuesta_api":"{'message':'No cumple criterio de antecedentes.}",
                                    "success":False}
                                   )
            return -2,js_aux
        elif not(cumple) and not(have_score_global):
            js_aux.update({"founded_cod":found_cods,
                                    "respuesta_api":None,
                                    "success":False}
                                   )
            return -3,js_aux
        else: # not(firmado):
            return -1,docs_pendientes
        

"""
Resultado API:
    200:
    {
    "success": true,
    "message": "Commerce created successfully",
    "data": {
        "success": true,
        "message": "Aliado creado exitosamente",
        "data": {
            "idComercio": 540345,
            "idTerminal": 383307,
            "url": "idkey=6870db382d96a648aeb12ce086ed049e&idus=cristianjimenez26"
        }
    }
    500:
    Internal Server Error
"""
        # print(f"Resuesta API:\n{response.text}")
        # return response
"""
- Por defecto Bogota, y reporte con alerta a equipo de Vinculación.
- Cotejamiento con lo introducido por el usuario, que lo de la geo concuerde con la ciudad y departamento en un porcentaje. 
- Recoleccion de informaciómn, Prospección vinculación (Registro PuntoRed)
- 
"""

# COMMAND ----------

def save_logs_to_table(js):
    if js:# and 'body' in js:
        body = js
        # Convert date strings to timestamps
        body['fecha_nacimiento'] = body.get('fecha_nacimiento', None)
        body['fecha_expedicion'] = body.get('fecha_expedicion', None)
        body['fecha'] = body.get('fecha', None)
        
        df = spark.createDataFrame([body])
        
        # Convert columns to timestamp
        df = df.withColumn("fecha_nacimiento", df["fecha_nacimiento"].cast("timestamp"))
        df = df.withColumn("fecha_expedicion", df["fecha_expedicion"].cast("timestamp"))
        df = df.withColumn("fecha", df["fecha"].cast("timestamp"))
        
        try:
            df.write.format("delta").mode("append").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn")
        except Exception as e:
            if 'Table or view not found' in str(e):
                df.write.format("delta").mode("overwrite").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn")
            else:
                raise e


def save_logs_to_table_segundo_flujo_promotor_pj(js):
    if js:# and 'body' in js:
        body = js
        # Convert date strings to timestamps
        body['fecha_nacimiento'] = body.get('fecha_nacimiento', None)
        body['fecha_expedicion'] = body.get('fecha_expedicion', None)
        body['fecha'] = body.get('fecha', None)
        
        df = spark.createDataFrame([body])
        
        # Convert columns to timestamp
        df = df.withColumn("fecha_nacimiento", df["fecha_nacimiento"].cast("timestamp"))
        df = df.withColumn("fecha_expedicion", df["fecha_expedicion"].cast("timestamp"))
        df = df.withColumn("fecha", df["fecha"].cast("timestamp"))
        
        try:
            df.write.format("delta").mode("append").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn_para_promotor")
        except Exception as e:
            if 'Table or view not found' in str(e):
                df.write.format("delta").mode("overwrite").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn_para_promotor")
            else:
                raise e


def save_logs_to_table_issues_background(js):
    if js:# and 'body' in js:
        body = js
        # Convert date strings to timestamps
        body['fecha_nacimiento'] = body.get('fecha_nacimiento', None)
        body['fecha_expedicion'] = body.get('fecha_expedicion', None)
        body['fecha'] = body.get('fecha', None)
        
        df = spark.createDataFrame([body])
        
        # Convert columns to timestamp
        df = df.withColumn("fecha_nacimiento", df["fecha_nacimiento"].cast("timestamp"))
        df = df.withColumn("fecha_expedicion", df["fecha_expedicion"].cast("timestamp"))
        df = df.withColumn("fecha", df["fecha"].cast("timestamp"))
        try:
            df.write.format("delta").mode("append").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn_issues_background")
        except Exception as e:
            if 'Table or view not found' in str(e):
                df.write.format("delta").mode("overwrite").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn_issues_background")

            else:
                raise e

def save_logs_to_table_tercer_flujo(js):
    if js:# and 'body' in js:
        body = js
        # Convert date strings to timestamps
        # body['fecha_nacimiento'] = body.get('fecha_nacimiento', None)
        # body['fecha_expedicion'] = body.get('fecha_expedicion', None)
        body['fecha'] = body.get('fecha', None)
        
        df = spark.createDataFrame([body])
        
        # Convert columns to timestamp
        # df = df.withColumn("fecha_nacimiento", df["fecha_nacimiento"].cast("timestamp"))
        # df = df.withColumn("fecha_expedicion", df["fecha_expedicion"].cast("timestamp"))
        df = df.withColumn("fecha", df["fecha"].cast("timestamp"))
        try:
            df.write.format("delta").mode("append").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_promotor_indirecto_pn")
        except Exception as e:
            if 'Table or view not found' in str(e):
                df.write.format("delta").mode("overwrite").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_promotor_indirecto_pn")

            else:
                raise e

def save_signio_gestion_vinculacion_tat_indirecto_pn(js):
    if js:# and 'body' in js:
        body = js
        # Convert date strings to timestamps
        # body['fecha_nacimiento'] = body.get('fecha_nacimiento', None)
        # body['fecha_expedicion'] = body.get('fecha_expedicion', None)
        # body['fecha'] = body.get('fecha', None)
        
        df = spark.createDataFrame([body])
        
        # Convert columns to timestamp
        # df = df.withColumn("fecha_nacimiento", df["fecha_nacimiento"].cast("timestamp"))
        # df = df.withColumn("fecha_expedicion", df["fecha_expedicion"].cast("timestamp"))
        # df = df.withColumn("fecha", df["fecha"].cast("timestamp"))
        try:
            df.write.format("delta").mode("append").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_signio_indirecto_pn_gestion")
        except Exception as e:
            if 'Table or view not found' in str(e):
                df.write.format("delta").mode("overwrite").saveAsTable("bronze.logs_whatsapp.flujo_whatsapp_signio_indirecto_pn_gestion")

            else:
                raise e

# COMMAND ----------

# DBTITLE 1,Procesos
try:
    files_procesados = (spark
                        .sql("""SELECT 
                                    DISTINCT file_logs_flows_whatsapp
                                    FROM
                                        silver.puntos_de_venta.vinculacion_flujo_whatsapp_truora_indirecto_pn
                            """)
                        .toPandas()["file_logs_flows_whatsapp"]
                        .tolist()
                        )
except Exception as e:
    print(f"Error: {e}")
    if 'TABLE_OR_VIEW_NOT_FOUND' in str(e):
        files_procesados = []

try:
    files_procesados_issues_background = (spark
                                        .sql("""SELECT 
                                                    DISTINCT file_logs_flows_whatsapp
                                                    FROM
                                                        bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn_issues_background
                                                    WHERE
                                                        respuesta_api NOT LIKE "%Internal Server Error%"

                                            """)
                                        .toPandas()["file_logs_flows_whatsapp"]
                                        .tolist()
                                        ) 
except:
    files_procesados_issues_background = []
# files_procesados = []
files_procesados += files_procesados_issues_background

folder = "vinculacion_indirecto"
bucket_name = 'logs-flows-whatsapp'
prefix_name_file = "vinculacion_indirecta"

template = f"{prefix_name_file}"

print(f"{bucket_name}--{folder}--{prefix_name_file}")
continuation_token = None
contents = []
while True:
    if continuation_token:
        list_docs = client.list_objects_v2(Bucket = bucket_name,
                                    Prefix = folder,
                                    ContinuationToken = continuation_token
                                    )    
    else:    
        list_docs = client.list_objects_v2(Bucket = bucket_name,
                                    Prefix = folder
                                    )
    if 'Contents' in list_docs:
        contents+= list_docs['Contents']


    if list_docs.get('IsTruncated'):
        continuation_token = list_docs['NextContinuationToken']
    else:
        break  # No hay más objetos
files = [content['Key'] for content in contents]


files = [file for file in files if template in file]
files = [file for file in files if file not in files_procesados]



# COMMAND ----------

create_obj_comercie = create_comercie(token_api_key = TOKEN_API_KEY,
                                      truora_api_key = TRUORA_API_KEY,
                                      client = client,
                                      signio_email=SIGNIO_EMAIL,
                                      signio_password=SIGNIO_PASSWORD
                                      )

intentos = 10
i=1
while True:
    res,res_api = activate_creation_comercie(TOKEN_API_KEY)
    if res and "pong" in res_api.text:
        break
    else:
        sleep(15)
    
    if i>intentos:
        send_message_slack(message = "No se enciende la infra para la creación del comercio.")
        break
    i+=1


failed_creacion = []
faltan_firmas = []
problemas_geo = []
issues_background = []
not_background = []
files_par = [file for file in files if "promotor" not in file]
for file in files_par:
    fecha = file.split(".")[0].split("_")[-6:]
    fecha = "-".join(fecha[0:3]) + " " +":".join(fecha[3:])
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
    threshold_date = datetime(2024, 9, 18,8,0,0)
    print(fecha_dt >= threshold_date)
    if fecha_dt >= threshold_date:
        
        js = create_obj_comercie.load_logs(file,"logs-flows-whatsapp")
        #print(js)
        if js["signed_by"] not in ["","null","flujo_externo_promotor"]:
            cod,respuesta = create_obj_comercie.create_comercie(js,df_mun)
            if cod == 1:
                if not(respuesta["founded_cod"]):
                    problemas_geo.append(pd.DataFrame(respuesta, index=[0]))
                respuesta.update({"file_logs_flows_whatsapp":file})
                save_logs_to_table(respuesta)

            if cod==0:
                if not(respuesta["founded_cod"]):
                    problemas_geo.append(pd.DataFrame(respuesta, index=[0]))
                respuesta.update({"file_logs_flows_whatsapp":file})
                save_logs_to_table_issues_background(respuesta)
                failed_creacion.append(pd.DataFrame(respuesta, index=[0]))
            if cod==-1:
                faltan_firmas.append(respuesta)
                
            if cod == -2:
                issues_background.append(pd.DataFrame(respuesta, index=[0]))
                respuesta.update({"file_logs_flows_whatsapp":file})
                save_logs_to_table_issues_background(respuesta)
            if cod == -3:
                not_background.append(pd.DataFrame(respuesta, index=[0]))

issues_background  = [issue_background for issue_background in issues_background  if issue_background.shape[1]!=0 ] 

# COMMAND ----------

if len(failed_creacion)>0:
    path = "/tmp/failed_creacion.xlsx"
    message = ":alert-siren: Error al crear comercios TAT INDIRECTO :alert-siren:"
    title = "Comercios No Creados"
    df_failed = pd.concat(failed_creacion,ignore_index=True)
    df_failed.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)


if len(problemas_geo)>0:
    path = "/tmp/problemas_geo_vinculacion.xlsx"
    message = ":alert-siren: :earth_americas: No se pudo validar por completo ciudad o departamento  TAT INDIRECTO :earth_americas: :alert-siren:"
    title = "Usuarios con problemas de Geo"
    df_problemas_geo = pd.concat(problemas_geo,ignore_index=True)
    df_problemas_geo.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)

if len(faltan_firmas)>0:
    df_faltan_firmas = faltan_firmas
    df_firmas = pd.concat(df_faltan_firmas,ignore_index=True)
    path = "/tmp/faltan_firmas.xlsx"
    message = ":alert-siren: Usuarios no han firmado todos los documentos TAT INDIRECTO :alert-siren:"
    title = "Usuarios que faltan por firmar documentos "
    df_firmas.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)


if len(issues_background)>0:
    path = "/tmp/problemas_check_background.xlsx"
    message = ":alert-siren: :shipit: No cumple  criterio de Score de Check Background TAT INDIRECTO :shipit: :alert-siren:"
    title = "Usuarios con antecedentes"
    df_check = pd.concat(issues_background,ignore_index=True)
    df_check.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)
    

if len(not_background)>0:
    path = "/tmp/no_check_background.xlsx"
    message = ":alert-siren: :alerta: No se encontraron consultas de antecedentes  TAT INDIRECTO :alerta: :alert-siren:"
    title = "Usuarios sin check background terminado"
    df_no_check = pd.concat(not_background,ignore_index=True)
    df_no_check.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)

# COMMAND ----------

# DBTITLE 1,Guardar  info flujos promotor.
files_ter_procesados = spark.sql("SELECT DISTINCT file FROM bronze.logs_whatsapp.flujo_whatsapp_promotor_indirecto_pn").toPandas()['file'].tolist()
files_tercer = [file for file in files if "promotor" in file]
files_tercer = [file for file in files_tercer if file not in files_ter_procesados]

for file in files_tercer:
    js = create_obj_comercie.load_logs(file,"logs-flows-whatsapp")
    js.update({"file":file})
    save_logs_to_table_tercer_flujo(js)

# COMMAND ----------

# DBTITLE 1,Envio sobre SIGNIO flujos externos
list_file_procesados = spark.sql("SELECT DISTINCT file_procesado FROM bronze.logs_whatsapp.flujo_whatsapp_signio_indirecto_pn_gestion").toPandas()['file_procesado'].tolist()
list_file_procesados2 = spark.sql("SELECT DISTINCT file_logs_flows_whatsapp FROM silver.puntos_de_venta.vinculacion_flujo_whatsapp_truora_indirecto_pn WHERE success").toPandas()['file_logs_flows_whatsapp'].tolist()
list_file_procesados += list_file_procesados2  
files_main = [file for file in files if "promotor" not in file]

files_main = [file for file in files_main if file not in list_file_procesados]

list_canales_mayo = [
                    "TAT GRANDES CUENTAS",
                    "OFICINA ALIADA",
                    "MAYORISTA",
                    "OF ALIADAS Y MAYORISTAS"
                    ]
for file in files_main:
    fecha = file.split(".")[0].split("_")[-6:]
    fecha = "-".join(fecha[0:3]) + " " + ":".join(fecha[3:])
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
    threshold_date = datetime(2025, 3, 7, 8, 0, 0)
    if fecha_dt >= threshold_date:

        js = create_obj_comercie.load_logs(file, "logs-flows-whatsapp").copy()
        # js = js__.copy()
        try:
            js_pro = spark.sql(
                f"""SELECT * FROM bronze.logs_whatsapp.flujo_whatsapp_promotor_indirecto_pn 
                WHERE id_proceso_flujo1 = '{js['id_proceso_flujo1']}'"""
            ).toPandas().iloc[0].to_dict()
        except:
            continue
        js_promotor = spark.sql(f"""SELECT
                                    *
                                    FROM
                                        silver.puntos_de_venta.distribuidor_vinculacion
                                    WHERE distribuidor_id = {js['id_promotor']}
                                """).toPandas().iloc[0].to_dict()
        name, lastname = create_obj_comercie.get_validation(js['id_proceso_flujo1'])
        event = {
        "nombre_participe":f"{name} {lastname}",
        "no_documentoparticipe":js['cedula'],
        "fecha_sobre": (datetime.utcnow() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'),
        "tipo_documento_participe": "CC" if js["tipo_documento"] == 'national-id' else "CE",
        "tipo_persona":"Natural",
        "razon_social": js['razon_social_aliado'],
        "nro_pagare":js['cedula'],
        "direccion_participe":js['direccion'],
        "ciudad_participe":js['ciudad'],
        "telefono_participe":js['telefono'].replace('+57',''),
        "email_participe":js["correo"],
        "nombre_promotor":js_promotor["nombre_mayorista"] if js_promotor["canal_comercial_distribuidor"] in list_canales_mayo else js_promotor["nombre_distribuidor"],
        "tipo_documento_promotor_letras":"Cedula de Ciudadanía" if js_pro["tipo_documento"] == 'CC' else "Cedula de Extrangería",
        "no_documentopromotor":js_pro["documento"],
        "tipo_persona_promotor":"Juridica",
        "razon_social_promotor":js_promotor["razon_social_mayorista"] if js_promotor["canal_comercial_distribuidor"] in list_canales_mayo else js_promotor["razon_social_distribuidor"],
        "tipo_documento_participe_letras":"Cedula de Ciudadanía" if js["tipo_documento"] == 'national-id' else "Cedula de Extrangería",
        "nit_razon_social":js_promotor["documento_mayorista"] if js_promotor["canal_comercial_distribuidor"] in list_canales_mayo else js_promotor["documento_distribuidor"],
        "email_promotor":js_promotor["email_mayorista"] if js_promotor["canal_comercial_distribuidor"] in list_canales_mayo else js_promotor["email_distribuidor"],
        "telefono_promotor":js_promotor["phone_mayorista"] if js_promotor["canal_comercial_distribuidor"] in list_canales_mayo else js_promotor["phone_distribuidor"],
        }
        response = api_sobre_3_partes_pn(event,SIGNIO_EMAIL,SIGNIO_PASSWORD)
        body = response['body']
        body["event"] = json.dumps(body["event"])
        body.pop("success")
        body.pop("mensaje")
        body['js'] = json.dumps(js)
        body['creado'] = False
        body["file_procesado"] = file
        save_signio_gestion_vinculacion_tat_indirecto_pn(body)
        






# COMMAND ----------

df_indirecto_pn = spark.sql("""   
select
  *
  from
    bronze.logs_whatsapp.flujo_whatsapp_signio_indirecto_pn_gestion
  where
    creado = false
    AND js:fecha::DATE > "2025-03-01" 
    AND js:signed_by =  "flujo_externo_promotor"
    -- IDP84efe8861d0d6a58960120c2d24492e3
""").toPandas()

# COMMAND ----------

display(df_indirecto_pn)

# COMMAND ----------

# DBTITLE 1,Creacion comercio  promotor Juridico
create_obj_comercie = create_comercie(token_api_key = TOKEN_API_KEY,
                                      truora_api_key = TRUORA_API_KEY,
                                      client = client,
                                      signio_email=SIGNIO_EMAIL,
                                      signio_password=SIGNIO_PASSWORD
                                      )
intentos = 10
i=1
while True:
    res,res_api = activate_creation_comercie(TOKEN_API_KEY)
    if res and "pong" in res_api.text:
        break
    else:
        sleep(15)
    
    if i>intentos:
        send_message_slack(message = "No se enciende la infra para la creación del comercio.")
        break
    i+=1


failed_creacion = []
faltan_firmas = []
problemas_geo = []
issues_background = []
not_background = []
for i,row in df_indirecto_pn.iterrows():
        js = json.loads(row["js"])
        # js['id_promotor'] = '259357'
        js.update({"id_documento_contrato":row['id_transaccion'],
                   "id_firma_electronica_usuario":row['id_firmante_participe'],
                   "id_firma_electronica_promotor":row['id_firmante_promotor']})
        # if js["signed_by"] not in ["","null","flujo_externo_promotor"]:
        print(row['file_procesado'])
        print("COD="+str(cod))
        cod,respuesta = create_obj_comercie.create_comercie(js,df_mun)
        
        if cod == 1:
                if not(respuesta["founded_cod"]):
                    problemas_geo.append(pd.DataFrame(respuesta, index=[0]))
                respuesta.update({"file_logs_flows_whatsapp":row['file_procesado']})
                save_logs_to_table(respuesta)
        
        if cod == 0:
            if not(respuesta["founded_cod"]):
                problemas_geo.append(pd.DataFrame(respuesta, index=[0]))
            respuesta.update({"file_logs_flows_whatsapp":row['file_procesado']})
            save_logs_to_table(respuesta)

        if cod==0:
            if not(respuesta["founded_cod"]):
                problemas_geo.append(pd.DataFrame(respuesta, index=[0]))
            respuesta.update({"file_logs_flows_whatsapp":row['file_procesado']})
            save_logs_to_table_issues_background(respuesta)
            failed_creacion.append(pd.DataFrame(respuesta, index=[0]))
        if cod==-1:
            faltan_firmas.append(respuesta)
            
        if cod == -2:
            issues_background.append(pd.DataFrame(respuesta, index=[0]))
            respuesta.update({"file_logs_flows_whatsapp":row['file_procesado']})
            save_logs_to_table_issues_background(respuesta)
        if cod == -3:
            not_background.append(pd.DataFrame(respuesta, index=[0]))

issues_background  = [issue_background for issue_background in issues_background  if issue_background.shape[1]!=0 ]

# COMMAND ----------

# DBTITLE 1,Envío Comunicaciones TAT Indirecto PN
if len(failed_creacion)>0:
    path = "/tmp/failed_creacion_tat_indirecto.xlsx"
    message = ":alert-siren: Error al crear comercios TAT Indirecto TAT INDIRECTO :alert-siren:"
    title = "Comercios No Creados TAT Indirecto"
    df_failed = pd.concat(failed_creacion,ignore_index=True)
    df_failed.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)


if len(problemas_geo)>0:
    path = "/tmp/problemas_geo_vinculacion_tat_indirecto.xlsx"
    message = ":alert-siren: :earth_americas: TAT Indirecto, No se pudo validar por completo ciudad o departamento TAT INDIRECTO :earth_americas: :alert-siren:"
    title = "Usuarios con problemas de Geo, TAT Indirecto"
    df_problemas_geo = pd.concat(problemas_geo,ignore_index=True)
    df_problemas_geo.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)

if len(faltan_firmas)>0:
    df_firmas = pd.concat(faltan_firmas)
    #  = pd.concat(df_faltan_firmas,ignore_index=True)
    path = "/tmp/faltan_firmas_tat_indirecto.xlsx"
    message = ":alert-siren: TAT Indirecto Usuarios no han firmado todos los documentos TAT INDIRECTO :alert-siren:"
    title = "Usuarios que faltan por firmar documentos, TAT Indirecto "
    df_firmas.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)


if len(issues_background)>0:
    path = "/tmp/problemas_check_background_tat_indirecto.xlsx"
    message = ":alert-siren: :shipit: TAT Indirecto, No cumple  criterio de Score de Check Background  TAT INDIRECTO:shipit: :alert-siren:"
    title = "Usuarios con antecedentes TAT Indirecto"
    df_check = pd.concat(issues_background,ignore_index=True)
    df_check.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)
    

if len(not_background)>0:
    path = "/tmp/no_check_background_tat_indirecto.xlsx"
    message = ":alert-siren: :alerta: TAT Indirecto, No se encontraron consultas de antecedentes TAT INDIRECTO :alerta: :alert-siren:"
    title = "Usuarios sin check background terminado TAT Indirecto"
    df_no_check = pd.concat(not_background,ignore_index=True)
    df_no_check.to_excel(path,index=False)
    send_message_slack(message = message,
                       title = title,
                       filepath = path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Puesta en spreadsheet
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.puntos_de_venta.vinculacion_flujo_whatsapp_truora_indirecto_pn
# MAGIC SELECT
# MAGIC   * EXCEPT (respuesta_api,foto_fachada,recibo_publico),
# MAGIC   SUBSTRING_INDEX(SUBSTRING_INDEX(direccion_ubicacion, 'query=', -1), ',', 1) AS latitud,
# MAGIC   SUBSTRING_INDEX(SUBSTRING_INDEX(direccion_ubicacion, 'query=', -1), ',', -1) AS longitud,
# MAGIC   CASE WHEN respuesta_api:data:data:idComercio IS NOT NULL THEN respuesta_api:data:data:idComercio ELSE respuesta_api:data:idComercio END AS idComercio,
# MAGIC   CASE WHEN respuesta_api:data:data:idTerminal IS NOT NULL THEN respuesta_api:data:data:idTerminal ELSE respuesta_api:data:idTerminal END AS idTerminal
# MAGIC FROM
# MAGIC   (SELECT * except (respuesta_api),replace(replace(replace(replace(respuesta_api, "True", "true"),"False","false"),'‘','"'),'’','"') AS respuesta_api from bronze.logs_whatsapp.flujo_whatsapp_tat_indirecto_pn) 

# COMMAND ----------

dp = spark \
    .sql("select * from silver.puntos_de_venta.vinculacion_flujo_whatsapp_truora_indirecto_pn where idComercio is not null") \
    .toPandas() \
    .astype({'fecha_nacimiento':str,
           'fecha':str,
           'fecha_expedicion':str
           })

# COMMAND ----------

import boto3
AWS_ACCESS_KEY = dbutils.secrets.get(scope="data_secrets", key="AWS_ACCESS_KEY")
AWS_SECRET_KEY = dbutils.secrets.get(scope="data_secrets", key="AWS_SECRET_KEY")
AWS_BUCKET = dbutils.secrets.get(scope="data_secrets", key="AWS_BUCKET")

session = boto3.session.Session()
client = session.client('s3', region_name='us-east-1', endpoint_url='https://s3.amazonaws.com',aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)

try:
    with open('/tmp/credentials_spreadsheets_2.json','wb') as f:
        client.download_fileobj( AWS_BUCKET, 'external-files/gspread/produccion-cxr-d153f40ff93d.json', f)
except Exception as e:
    print('Exception: ', e)
    pass

# COMMAND ----------

# DBTITLE 1,Write spreadsheet
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd

#Compartir el archivo con este correo credencials-drive-api@produccion-cxr.iam.gserviceaccount.com
# Ruta al archivo JSON de credenciales de la cuenta de servicio
ruta_credenciales = '/tmp/credentials_spreadsheets_2.json'

# Alcance de la API de Google Sheets
alcance_sheets = ['https://www.googleapis.com/auth/spreadsheets']

# Crea un objeto de servicio para la API de Google Sheets
servicio_sheets = build('sheets', 'v4', credentials=service_account.Credentials.from_service_account_file(ruta_credenciales, scopes=alcance_sheets))

# ID de la hoja de cálculo de Google Sheets
id_hoja_calculo = '1viaM8Dc-_CfSTyWNUVx81l-4OUWCPXco0x6PoUihM-8'

# Nombre del rango que deseas leer
# hoja = 'Nueva caida de info!A:K'

# Ejemplo de lectura de datos
# resultado = servicio_sheets.spreadsheets().values().get(spreadsheetId=id_hoja_calculo, range=hoja).execute()
# datos = resultado.get('values', [])

# columnas = datos[0]  # La primera fila se usa como nombres de columnas
# datos = datos[1:]  # Resto de las filas son datos

# dp = pd.DataFrame(datos, columns=columnas)

# # Imprime el DataFrame
# display(dp)


# Nombre de la hoja en la que deseas escribir
nombre_hoja = 'Indirecto'

#Rango de celdas a agregar data
start_row = 1
end_row = start_row + len(dp)

nC = dp.shape[1]
columns_base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
res = nC%len(columns_base)
num = int(nC/len(columns_base))

last_col = columns_base[nC-1] if num == 0 else f"{columns_base[num-1]}{columns_base[res-1]}"
# Nombre del rango en el que deseas escribir (puedes ajustar según tu necesidad)
nombre_rango = f'{nombre_hoja}!A{start_row}:{last_col}{end_row}'  # Toda la data
# nombre_rango = f'{nombre_hoja}!L{start_row}:P{end_row}' # Campos que valido

# Convierte el DataFrame en una lista de listas para la escritura en Sheets
# datos_lista = [dp.columns.tolist()] + dp.values.tolist()
# dpn = dp[['pdv','id_hubspot_pdv','onboarding','id_hubspot_onboarding','etapa']]
datos_lista = [dp.columns.tolist()] + dp.values.tolist() # Campos que valido

# Prepara la solicitud para escribir en la hoja de cálculo
solicitud_escritura = servicio_sheets.spreadsheets().values().update(
    spreadsheetId=id_hoja_calculo,
    range=nombre_rango,
    valueInputOption='RAW',
    body={'values': datos_lista}
)

# Ejecuta la solicitud
respuesta = solicitud_escritura.execute()

print("Datos escritos exitosamente en la hoja de cálculo.")

# COMMAND ----------

# MAGIC %md
# MAGIC # consultas background

# COMMAND ----------

# df = spark.sql("""
#                SELECT
#                 *
#                 FROM 
#                   bronze.logs_whatsapp.flujo_whatsapp_tat_directo_pn
#                """).toPandas()

# def get_check_background_results(check_id):
#     ENDPOINT = f"https://api.checks.truora.com/v1/checks/{check_id}"
#     headers = {"Truora-API-Key": TRUORA_API_KEY}
#     response = requests.get(ENDPOINT, headers=headers)
#     return response.text

# create_obj_comercie = create_comercie(token_api_key = TOKEN_API_KEY,
#                                       truora_api_key = TRUORA_API_KEY,
#                                       client = client,
#                                       signio_email=SIGNIO_EMAIL,
#                                       signio_password=SIGNIO_PASSWORD
#                                       )

# background_response = []
# for i,row in df.iterrows():
#     # background_response.append(get_check_background_results(row['check_id']))
#     check_id = create_obj_comercie.get_check_id(row["id_proceso_flujo1"])
#     response = get_check_background_results(check_id) if check_id else "{'message':'not found}"
#     background_response.append(response)



# df_aux = df[["cedula",
#              "telefono",
#              "id_proceso_flujo1",
#              "id_documento_contrato",
#              "id_firma_electronica_usuario"]]
# df_aux["check_json"] = background_response


# df_aux_spark = spark.createDataFrame(df_aux)
# df_aux_spark.write.format("delta").mode("overwrite").saveAsTable("silver.default.detalle_background")

# COMMAND ----------

# %sql
# WITH json AS (
# SELECT
# cedula
# ,telefono
# ,id_proceso_flujo1
# ,id_documento_contrato
# ,id_firma_electronica_usuario
# ,from_json(check_json,
#           'struct<
#             check: struct<
#               check_id: string,
#               company_summary: struct<
#                 company_status: string,
#                 result: string
#               >,
#               country: string,
#               creation_date: string,
#               date_of_birth: string,
#               document_recognition_id: string,
#               expedition_date: string,
#               issue_date: string,
#               first_name: string,
#               name_score: float,
#               id_score: float,
#               last_name: string,
#               score: float,
#               scores: array<struct<
#                 data_set: string,
#                 severity: string,
#                 score: float,
#                 result: string,
#                 by_id: struct<
#                   result: string,
#                   score: float,
#                   severity: string
#                 >,
#                 by_name: struct<
#                   result: string,
#                   score: float,
#                   severity: string
#                 >
#               >>,
#               status: string,
#               statuses: array<struct<
#                 database_id: string,
#                 database_name: string,
#                 data_set: string,
#                 status: string,
#                 invalid_inputs: array<string>
#               >>,
#               summary: struct<
#                 gender: string,
#                 identity_status: string,
#                 names_found: array<struct<
#                   first_name: string,
#                   last_name: string,
#                   count: int
#                 >>,
#                 result: string
#               >,
#               update_date: string,
#               vehicle_summary: struct<
#                 result: string,
#                 vehicle_status: string
#               >,
#               national_id: string,
#               type: string
#             >,
#             details: string,
#             self: string
#           >
#           ' ) AS  json_struct
#   FROM
#     silver.default.detalle_background
# )
# select
#   json.* EXCEPT (json_struct)
#   ,m.data_set
#   ,m.severity
#   ,m.score
#   ,m.result
#   ,m.by_id.result AS by_id_result
#   ,m.by_id.score AS by_id_score
#   ,m.by_id.severity AS by_id_severity
#   ,m.by_name.result AS by_id_result
#   ,m.by_name.score AS by_id_score
#   ,m.by_name.severity AS by_id_severity
  
#   from json
#   lateral view explode(json.json_struct.check.scores) as m

# COMMAND ----------


