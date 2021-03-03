from owslib.fes import PropertyIsLike
from owslib.etree import etree
from owslib.wfs import WebFeatureService
import xml.etree.ElementTree as ET
import urllib


webservers = ['https://gs.geoscience.nsw.gov.au/geoserver/wfs', 'http://geology.information.qld.gov.au/geoserver/wfs', 'http://sarigdata.pir.sa.gov.au/geoserver/wfs', 'http://geossdi.dmp.wa.gov.au/services/wfs', 'http://www.mrt.tas.gov.au/web-services/wfs', 'http://geology.data.nt.gov.au/geoserver/wfs', 'http://geology.data.vic.gov.au/nvcl/wfs']

nvcldataservicespath = '/NVCLDataServices'

for webserver in webservers:
  wfs11 = WebFeatureService(url=webserver, version='1.1.0')

  filter = PropertyIsLike(propertyname='nvclCollection', literal='true', wildCard='*')
  filterxml = etree.tostring(filter.toXML()).decode("utf-8")
  response = wfs11.getfeature(typename='gsmlp:BoreholeView', filter=filterxml)

  root = ET.fromstring(response.read())

  print('Boreholes and associated datasets for service:' + webserver)
  print('Borehole ID, DatasetName, DatasetID')

  for child in root.findall('./*/{http://xmlns.geosciml.org/geosciml-portrayal/4.0}BoreholeView'):
    nvcl_id = child.attrib.get('{http://www.opengis.net/gml}id', '').split('.')[-1:][0]
    params = {'holeidentifier' : nvcl_id}
    enc_params = urllib.parse.urlencode(params).encode('ascii')
    req = urllib.request.Request(webserver.rpartition('/')[0].rpartition('/')[0]+nvcldataservicespath+'/getDatasetCollection.html', enc_params)
    with urllib.request.urlopen(req, timeout=60) as NVCLDSresponse:
      NVCLroot = ET.fromstring(NVCLDSresponse.read())
      datasets = NVCLroot.findall("./Dataset")
      if len(datasets) ==0 :
        print(nvcl_id + ', NONE, NONE' ) 
      else :
        for dataset in datasets:
          datasetid = dataset.find("DatasetID").text
          datasetname = dataset.find("DatasetName").text
          print(nvcl_id + ', ' + datasetname + ', ' + datasetid)
  print('END Boreholes and associated datasets for service:' + webserver)
  print('')
  print('')
      
        
