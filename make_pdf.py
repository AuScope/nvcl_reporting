import os
from fpdf import FPDF
from PIL import Image

REPORT_FILE = 'report.pdf'
IMAGE_DIR = 'outputs'
IMAGE_SZ = [150, 100]

# PDF class used to customize page layout
class PDF(FPDF):
    # Page header
    def header(self):
        # Insert AuScope logo
        self.image(os.path.join('images','AuScope.png'), 10, 8, 33)
        # Set font to helvetica bold 15
        self.set_font('helvetica', 'B', 15)
        # Move to the right
        self.cell(80)
        # Make title
        self.cell(30, 10, 'NVCL Report', 'B', 0, 'C')
        # Insert line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Set font to helvetica italic 8
        self.set_font('helvetica', 'I', 8)
        # Write page number
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

#
# Main part starts here
#
report_sections = { 'Element Graphs': [ 'elems_count.png', 'elems_state.png', 'elem_suffix_stats.png', 'elem_S.png',
                               'elems_suffix.png'],
           'Geophysics Graphs': [ 'geophys_count.png', 'geophys_state.png' ],
           'Boreholes Graphs': [ 'log1_geology.png', 'log1_nonstdalgos.png' ]
         }

# Create an A4 portrait PDF file
pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.add_page()
pdf.set_font('Times', '', 12)

link_list = []
for section_header in report_sections:
    link_id = pdf.add_link()
    link_list.append(link_id)
    pdf.cell(0, 10, section_header, 0, 1, link=link_id)
    
pdf.add_page()


# Iterate over sections
for idx, (section_header, image_list) in enumerate(report_sections.items()):
    pdf.cell(0, 10, section_header, 0, 1)
    pdf.set_link(link_list[idx])
    # Iterate over images within each section
    for image in image_list: 
        image_file = os.path.join(IMAGE_DIR, image)
        with Image.open(image_file) as img:
            # Resize image without changing aspect ratio
            src_aspect = img.size[0]/img.size[1]
            dest_aspect = IMAGE_SZ[0]/IMAGE_SZ[1]
            if src_aspect > dest_aspect:
                # Wider
                out_w = IMAGE_SZ[0]
                out_h = IMAGE_SZ[1] * dest_aspect / src_aspect
            else:
                # Taller
                out_w = IMAGE_SZ[0] * src_aspect / dest_aspect
                out_h = IMAGE_SZ[1]
            pdf.image(image_file, w=out_w, h=out_h)
    pdf.add_page()

# Create report file
if os.path.exists(REPORT_FILE):
    os.remove(REPORT_FILE)
pdf.output(REPORT_FILE)
