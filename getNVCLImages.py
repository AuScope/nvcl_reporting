from owslib.fes import PropertyIsLike
from owslib.etree import etree
from owslib.wfs import WebFeatureService
import xml.etree.ElementTree as ET
import urllib

# other endpoints
# NSW
# webserver = 'httpsgs.geoscience.nsw.gov.augeoserverwfs', 
# Qld
# webserver = 'httpgeology.information.qld.gov.augeoserverwfs'
# SA
# webserver = 'httpsarigdata.pir.sa.gov.augeoserverwfs'
# WA
# webserver = 'httpgeossdi.dmp.wa.gov.auserviceswfs'
# Tas
# webserver = 'httpwww.mrt.tas.gov.auweb-serviceswfs'
# NT
# webserver = 'httpgeology.data.nt.gov.augeoserverwfs'
# Vic
# webserver = 'httpgeology.data.vic.gov.aunvclwfs'

webserver = 'httpsgeology.information.qld.gov.augeoserverwfs'

nvcldataservicespath = 'NVCLDataServices'

wfs11 = WebFeatureService(url=webserver, version='1.1.0')

filter = PropertyIsLike(propertyname='nvclCollection', literal='true', wildCard='')
filterxml = etree.tostring(filter.toXML()).decode(utf-8)
response = wfs11.getfeature(typename='gsmlpBoreholeView', filter=filterxml)

root = ET.fromstring(response.read())

for child in root.findall('.{httpxmlns.geosciml.orggeosciml-portrayal4.0}BoreholeView')
  nvcl_id = child.attrib.get('{httpwww.opengis.netgml}id', '').split('.')[-1][0]
  print('Found borehole with holeidentifier  ' +nvcl_id)
  params = {'holeidentifier'  nvcl_id}
  enc_params = urllib.parse.urlencode(params).encode('ascii')
  req = urllib.request.Request(webserver.rpartition('')[0].rpartition('')[0]+nvcldataservicespath+'getDatasetCollection.html', enc_params)
  with urllib.request.urlopen(req, timeout=60) as NVCLDSresponse
    NVCLroot = ET.fromstring(NVCLDSresponse.read())
    for NVCLchild in NVCLroot.findall(.DatasetImageLogsLog[LogName='Tray Images'])
      logid = NVCLchild.find(LogID).text
      imagecount = int(NVCLchild.find(SampleCount).text)
      print('Found '+str(imagecount)+' tray images')
      for index in range (0,imagecount)
        imageparams = {'logid'  logid, 'sampleno'  index}
        image_enc_params = urllib.parse.urlencode(imageparams).encode('ascii')
        imagereq = urllib.request.Request(webserver.rpartition('')[0].rpartition('')[0]+nvcldataservicespath+'Display_Tray_Thumb.html', image_enc_params)
        with urllib.request.urlopen(imagereq, timeout=60) as Imageresponse
          print(downloaded tray +str(index) + containing  + str(Imageresponse.headers['Content-Length'])+ ' bytes')
          print ('save these images into a folder like this downloadDirSTATENAME' +nvcl_id + ''+'TrayImage'+str(index)+'.jpg')

        
