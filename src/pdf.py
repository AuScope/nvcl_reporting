import os
import glob
from fpdf import FPDF
from PIL import Image

from constants import IMAGE_SZ, FONT, ROOT_PATH
from report_table_data import ReportTableData


# PDF class used to customize page layout
class PDF(FPDF):

    def __init__(self, orientation="portrait", unit="mm", format="A4", header_title="NVCL Report"):
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.header_title = header_title

    # Page header
    def header(self):
        """ Write page header
        """
        # Insert AuScope logo
        img_path = os.path.join(ROOT_PATH, 'assets', 'images', 'AuScope.png')
        if os.path.isfile(img_path):
            self.image(img_path, 10, 8, 33)
        else:
            print(f"WARNING: AuScope logo {img_path} cannot be found, will be missing from report")

        # Set font to helvetica bold 15
        self.set_font(FONT, 'B', 15)
        # Move to the right
        self.cell(80)
        # Make title
        self.cell(w=30, h=10, text=self.header_title, border='B', align='C')
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
        self.cell(h=10, text=f"Page {self.page_no()}", align='C')

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
    pdf.cell(w=page_width, h=0.0, text=title, align='C')
    # Set table font
    pdf.set_font(FONT,'',10.0)
    # Line break before table
    pdf.ln(3.0)
    # Draw table
    for row in row_data:
        # Skip empty row
        if len(row) < 1:
            continue
        if col_width is None:
            col_width = page_width/len(row)
        for datum in row:
            if isinstance(datum, float):
                pdf.cell(w=col_width, h=2*text_height, text=f"{datum:.1f}", border=1)
            else:
                pdf.cell(w=col_width, h=2*text_height, text=str(datum), border=1)
        # Separator
        pdf.ln(2*text_height)


def write_report(report_file, image_dir, report: ReportTableData, metadata, brief):
    """ Writes a PDF report to filsystem

    :param report_file: path and filename of PDF report
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
        # Find all the elements graphs
        elem_graph_paths = glob.glob(os.path.join(image_dir, "elems_count_*.png"))
        elems_suffix_paths = glob.glob(os.path.join(image_dir, "elems_suffix_*.png"))
        elem_graphs = [os.path.basename(path) for path in elem_graph_paths]
        graph_sections = { 'Element Graphs': elem_graphs + [ 'elems_prov.png',
                                            'elem_suffix_stats.png', 'elem_S.png'] + elems_suffix_paths,
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
    pdf.multi_cell(w=0, h=pdf.font_size * 1.5, text="Contents\n")
    pdf.set_font('Times', '', 12)
    link_list = []
    for section_header in graph_sections:
        link_id = pdf.add_link()
        link_list.append(link_id)
        pdf.multi_cell(w=0, h=pdf.font_size * 1.2, text=section_header+"\n", link=link_id)

    # Write out report metadata
    pdf.set_font('Times', 'B', 14)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.5, text="Information\n")
    pdf.set_font('Times', '', 12)
    for key, val in metadata.items():
        print(f"Writing {key}: {val}")
        pdf.multi_cell(w=0, h=pdf.font_size * 1.2, text=f"{key}: {val}\n", align="L")
    
    pdf.add_page()

    # Lay out graphs: iterate over graph sections
    for idx, (section_header, image_list) in enumerate(graph_sections.items()):
        pdf.cell(h=10, text=section_header)
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
                pdf.ln()
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
