import os
from fpdf import FPDF
from PIL import Image

from constants import IMAGE_SZ, FONT
from report_table_data import ReportTableData


# PDF class used to customize page layout
class PDF(FPDF):

    def __init__(self, orientation="portrait", unit="mm", format="A4", font_cache_dir=True, header_title="NVCL Report"):
        super().__init__(orientation, unit, format, font_cache_dir)
        self.header_title = header_title

    # Page header
    def header(self):
        """ Write page header
        """
        # Insert AuScope logo
        img_path = os.path.join('assets', 'images', 'AuScope.png')
        if os.path.isfile(img_path):
            self.image(img_path, 10, 8, 33)
        else:
            print(f"WARNING: AuScope logo {img_path} cannot be found, will be missing from report")

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
        """ Write page footer
        """
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Set font to helvetica italic 8
        self.set_font(FONT, 'I', 8)
        # Write page number
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

def write_table(pdf: PDF, title: str, row_data: list):
    """ Write a table using a PDF class

    :param pdf: PDF() object to write to
    :param title: table's title
    :param data: row data
    """
    # Table column width
    col_width = None
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
    for row in row_data:
        if col_width is None:
            col_width = page_width/len(row)
        for datum in row:
            if isinstance(datum, float):
                pdf.cell(col_width, 2*text_height, f"{datum:.1f}", border=1)
            else:
                pdf.cell(col_width, 2*text_height, str(datum), border=1)
        pdf.ln(2*text_height)


def write_report(report_file, image_dir, report: ReportTableData, metadata, brief):
    """ Writes a PDF report to filsystem

    :param report_file: filename of PDF report
    :param image_dir: directory where it expects the images to be
    :param report: report table structure and values
    :param metadata: report metadata
    :param brief: iff True will do a brief report
    """

    # Define which graphs appear in which sections
    if brief:
        graph_sections = { 'Borehole Graphs': [ 'borehole_number.png', 'borehole_kilometres.png', 'borehole_number_q.png', 'borehole_number_y.png',
                             'borehole_kilometres_q.png', 'borehole_kilometres_y.png'  ]
        }
    else:
        graph_sections = { 'Element Graphs': [ 'elems_count.png', 'elems_prov.png',
                                            'elem_suffix_stats.png', 'elem_S.png',
                                            'elems_suffix.png'],
           'Geophysics Graphs': [ 'geophys_count.png', 'geophys_prov.png' ],
           'Borehole Graphs': [ 'borehole_number.png', 'borehole_kilometres.png', 'log1_geology.png', 'log1_nonstdalgos.png' ]
        }

    # Write out title page
    if brief:
        header_title="Brief NVCL Report"
    else:
        header_title="NVCL Report"
    # Create an A4 portrait PDF file
    pdf = PDF(orientation="P", unit="mm", format="A4", header_title=header_title)
    pdf.add_page()

    # Write out contents page
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, "Contents", 0, 1)
    pdf.set_font('Times', '', 12)
    link_list = []
    for section_header in graph_sections:
        link_id = pdf.add_link()
        link_list.append(link_id)
        pdf.cell(0, 12, section_header, 0, 1, link=link_id)

    # Write out report metadata
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 14, "Information", 0, 1)
    pdf.set_font('Times', '', 12)
    for key, val in metadata.items():
        pdf.multi_cell(w=0, h=12, txt=f"{key}: {val}", ln=1)
    
    pdf.add_page()

    # Lay out graphs: iterate over graph sections
    for idx, (section_header, image_list) in enumerate(graph_sections.items()):
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
        # One page per section
        pdf.add_page()

    # Lay out tables, four to a page
    for idx, tabl in enumerate(report.table_list):
        if idx % 4 == 0 and idx > 0:
            pdf.add_page()
        write_table(pdf, tabl.title, tabl.rows)

    # Write report file to filesystem
    if os.path.exists(report_file):
        os.remove(report_file)
    pdf.output(report_file)
