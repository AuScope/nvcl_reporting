from types import SimpleNamespace

class ReportTableData:
    """ Simple class to hold report table data
    """

    def __init__(self):
        """ Initialise class
        """
        # List of SimpleNamespace objects, each one is a table. Keys are: "title" and "rows"
        self.table_list = []

    def add_table(self, table_header: list, table_datarows: list, title: str):
        """ Adds table data structures for passing to report building routines

        :param table_header: list of table header strings
        :param table_datarows: list of data rows to be inserted into table, each row same length as headers string list
        :param title: title of data table
        """
        table_obj = SimpleNamespace()
        table_obj.title = title
        table_obj.rows = [table_header]
        table_obj.rows.append(table_datarows)
        self.table_list.append(table_obj)



