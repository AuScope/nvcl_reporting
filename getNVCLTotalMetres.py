from owslib.fes import PropertyIsLike
from owslib.etree import etree
from owslib.wfs import WebFeatureService
import xml.etree.ElementTree as ET
import urllib


webservers = [
  #'httpsgs.geoscience.nsw.gov.augeoserverwfs',
  'httpgeology.information.qld.gov.augeoserverwfs', 
  # 'httpssarigdata.pir.sa.gov.augeoserverwfs', 
  #'httpgeossdi.dmp.wa.gov.auserviceswfs', 
  #'httpwww.mrt.tas.gov.auweb-serviceswfs', 
  #'httpgeology.data.nt.gov.augeoserverwfs', 
  #'httpgeology.data.vic.gov.aunvclwfs'
  ]

nvcldataservicespath = 'NVCLDataServices'

for webserver in webservers
  wfs11 = WebFeatureService(url=webserver, version='1.1.0')

  filter = PropertyIsLike(propertyname='nvclCollection', literal='true', wildCard='', matchCase=False)
  filterxml = etree.tostring(filter.toXML()).decode(utf-8)
  response = wfs11.getfeature(typename='gsmlpBoreholeView', filter=filterxml)

  root = ET.fromstring(response.read())

  print('Boreholes and associated datasets for service' + webserver)
  print('Borehole ID, DatasetName, DatasetID')

  for child in root.findall('.{httpxmlns.geosciml.orggeosciml-portrayal4.0}BoreholeView')
    nvcl_id = child.attrib.get('{httpwww.opengis.netgml}id', '').split('.')[-1][0]
    params = {'holeidentifier'  nvcl_id}
    enc_params = urllib.parse.urlencode(params).encode('ascii')
    req = urllib.request.Request(webserver.rpartition('')[0].rpartition('')[0]+nvcldataservicespath+'getDatasetCollection.html', enc_params)
    with urllib.request.urlopen(req, timeout=60) as NVCLDSresponse
      NVCLroot = ET.fromstring(NVCLDSresponse.read())
      datasets = NVCLroot.findall(.Dataset)
      if len(datasets) ==0 
        print(nvcl_id + ', NONE, NONE' ) 
      else 
        for dataset in datasets
          logid = dataset.find(.LogsLogLogID).text
          domparam = [{'logid'logid},{'outputformat'  'json'}]
          dom_enc_params = urllib.parse.urlencode(domparam).encode('ascii')
          # domreq = urllib.request.Request(webserver.rpartition('')[0].rpartition('')[0]+nvcldataservicespath+'downloadscalars.html', enc_params)
          print(nvcl_id + ', ' + logid )
  print('END Boreholes and associated datasets for service' + webserver)
  print('')
  print('')
      
        
