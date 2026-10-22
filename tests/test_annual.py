from pathlib import Path
import pandas as pd
from siniestros_riesgo.annual import clean_annual_file


def test_annual_victim_sums(tmp_path: Path) -> None:
    columns={"geo_id":"CVEGEO","state_code":"CVE_ENT","municipality_code":"CVE_MUN","year":"ANIO","month":"MES","day":"ID_DIA","hour":"ID_HORA","weekday":"DIASEMANA","status":"ESTATUS","crash_class":"CLASACC","death_fields":["CONDMUERTO","PASAMUERTO","PEATMUERTO","CICLMUERTO","OTROMUERTO","NEMUERTO"],"injury_fields":["CONDHERIDO","PASAHERIDO","PEATHERIDO","CICLHERIDO","OTROHERIDO","NEHERIDO"]}
    row={name:0 for name in columns["death_fields"]+columns["injury_fields"]}; row.update({"CVEGEO":"01001","CVE_ENT":"01","CVE_MUN":"001","ANIO":2020,"MES":1,"ID_DIA":2,"ID_HORA":21,"DIASEMANA":"Sábado","ESTATUS":"Definitiva","CLASACC":"No fatal","CONDHERIDO":2})
    path=tmp_path/"annual.csv"; pd.DataFrame([row]).to_csv(path,index=False)
    result=clean_annual_file(path,2020,columns)
    assert result.loc[0,"injuries_total"] == 2
    assert result.loc[0,"victimas"] == 1
    assert result.loc[0,"night"] == 1
    assert result.loc[0,"weekend"] == 1
