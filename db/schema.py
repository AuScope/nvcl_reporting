'''
This is the schema for the database
'''
from peewee import Field, Model, TextField, DateField, CompositeKey

DATE_FMT = '%Y/%m/%d'
OLD_DATE_FMT = '%Y-%m-%d'

class ScalarsField(Field):
    field_type = 'Scalars'

    def db_value(self, data: any) -> str:
        '''
        Convert mineral types at each depth to JSON formatted string
        i.e. ((depth, {key: val ...}, ...) ... ) or NaN

        :param data: mineral types at each depth
        :returns: JSON formatted string
        '''
        data_list = []
        if isinstance(data, OrderedDict):
            for depth, obj in data.items():
                # vars() converts Namespace -> dict
                if isinstance(obj, SimpleNamespace):
                    data_list.append([depth, vars(obj)])
                elif isinstance(obj, list) and len(obj) == 0:
                    continue
                else:
                    print(repr(obj), type(obj))
                    print("ERROR Unknown obj type in 'data' var")
                    sys.exit(1)
        elif data != {} and data != [] and not isinstance(data, list) and not (isinstance(data, float) and math.isnan(data)):
            print(repr(data), type(data))
            print("ERROR Unknown type in 'data' var")
            sys.exit(1)

        return json.dumps(data_list)

    def python_value(self, value: str) -> any:
        ''' 
        Converts from JSON string to Python object

        :param value: JSON string
        :returns: object or [] upon error
        '''
        #try:
        return json.loads(value) # Convert JSON string to python obj
        #except json.decoder.JSONDecodeError:
        #    return []



class JSONField(Field):
    field_type = 'JSON'

    def db_value(self, values: any) -> str:
        '''
        Converts lists of values to JSON formatted string

        :param values: list of valuex
        :returns: JSON formatted string
        '''
        # Sometimes 'values' is not an iterable numpy array
        if isinstance(values, Iterable):
            value_out = json.dumps(list(values))
        elif isinstance(values, float) and math.isnan(values):
            value_out = '[]'
        else:
            value_out = json.dumps([values])
        return value_out

    def python_value(self, value: str) -> any:
        ''' 
        Converts from JSON string to Python object

        :param value: JSON string
        :returns: object or [] upon error
        '''
        #try:
        return json.loads(value) # Convert JSON string to python obj
        #except json.decoder.JSONDecodeError:
        #    return []


class Meas(Model):
    # ['report_category', 'provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data']
    report_category = TextField() # Can be any one of 'log1', 'log2', 'log6', 'empty' and 'nodata'
    provider = TextField()
    nvcl_id = TextField()
    modified_datetime = DateField(formats=[DATE_FMT]) # Only some providers supply it, else retrieval date is used
    log_id = TextField()
    algorithm = TextField()
    log_type = TextField()
    algorithm_id = TextField()
    minerals = JSONField() # Unique minerals
    mincnts = JSONField()  # Counts of unique minerals as an array
    data = ScalarsField()     # Mineral scalars data 

    class Meta:
        primary_key = CompositeKey('report_category', 'provider', 'nvcl_id', 'log_id', 'algorithm', 'log_type', 'algorithm_id')



