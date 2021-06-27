import os
from fpdf import FPDF
from PIL import Image

# Plot image size
IMAGE_SZ = [150, 100]
FONT = 'helvetica'

# PDF class used to customize page layout
class PDF(FPDF):

    def __init__(self, orientation="portrait", unit="mm", format="A4", font_cache_dir=True, header_title="NVCL Report"):
        super().__init__(orientation, unit, format, font_cache_dir)
        self.header_title = header_title

    # Page header
    def header(self):
        # Insert AuScope logo
        self.image(os.path.join('images','AuScope.png'), 10, 8, 33)
        # Set font to helvetica bold 15
        self.set_font(FONT, 'B', 15)
        # Move to the right
        self.cell(80)
        # Make title
        self.cell(30, 10, self.header_title, 'B', 0, 'C')
        # Insert line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Set font to helvetica italic 8
        self.set_font(FONT, 'I', 8)
        # Write page number
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

def write_table(pdf, title, data, new_page):
    # Table column width
    col_width = None
    # Add new page
    if new_page:
        pdf.add_page()
    # Page width
    page_width = pdf.w - 2*pdf.l_margin
    # Text height
    text_height = pdf.font_size
    # Line break
    pdf.ln(4*text_height)
    # Set table title font
    pdf.set_font(FONT,'B',14.0)
    # Create table title
    pdf.cell(page_width, 0.0, title, align='C')
    # Set table font
    pdf.set_font(FONT,'',10.0)
    # Line break before table
    pdf.ln(3.0)
    # Draw table
    for row in data:
        if col_width is None:
            col_width = page_width/len(row)
        for datum in row:
            if isinstance(datum, float):
                pdf.cell(col_width, 2*text_height, f"{datum:.1f}", border=1)
            else:
                pdf.cell(col_width, 2*text_height, str(datum), border=1)
        pdf.ln(2*text_height)


def write_report(report_file='report.pdf', image_dir='outputs', table_data=[], title_list=[], metadata={}, brief=True):

    #
    # Main part starts here
    #
    if brief:
        report_sections = { 'Boreholes Graphs': [ 'borehole_number.png', 'borehole_kilometres.png', 'borehole_number_q.png', 'borehole_number_y.png', 
                             'borehole_kilometres_q.png', 'borehole_kilometres_y.png'  ]
        }
    else:
        report_sections = { 'Element Graphs': [ 'elems_count.png', 'elems_state.png',
                                            'elem_suffix_stats.png', 'elem_S.png',
                                            'elems_suffix.png'],
           'Geophysics Graphs': [ 'geophys_count.png', 'geophys_state.png' ],
           'Boreholes Graphs': [ 'borehole_number.png', 'borehole_kilometres.png', 'log1_geology.png', 'log1_nonstdalgos.png' ]
        }

    # Create an A4 portrait PDF file
    if brief:
        header_title="Brief NVCL Report"
    else:
        header_title="NVCL Report"
    pdf = PDF(orientation="P", unit="mm", format="A4", header_title=header_title)
    pdf.add_page()
    pdf.set_font('Times', 'B', 14)

    pdf.cell(0, 10, "Contents", 0, 1)
    pdf.set_font('Times', '', 12)
    link_list = []
    for section_header in report_sections:
        link_id = pdf.add_link()
        link_list.append(link_id)
        pdf.cell(0, 12, section_header, 0, 1, link=link_id)

    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 14, "Information", 0, 1)
    pdf.set_font('Times', '', 12)
    for key, val in metadata.items():
        pdf.cell(w=0, h=12, txt=f"{key}: {val}", ln=1)
    
    pdf.add_page()

    # Iterate over sections
    for idx, (section_header, image_list) in enumerate(report_sections.items()):
        pdf.cell(0, 10, section_header, 0, 1)
        pdf.set_link(link_list[idx])
        # Iterate over images within each section
        for image in image_list: 
            image_file = os.path.join(image_dir, image)
            if not os.path.exists(image_file):
                print(f"WARNING: {image_file} cannot be found, will be missing from report")
                continue
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

    # Write tables, four to a page
    for idx, title in enumerate(title_list):
        write_table(pdf, title, table_data[idx], (idx + 1) % 4 == 0)

    # Create report file
    if os.path.exists(report_file):
        os.remove(report_file)
    pdf.output(report_file)
