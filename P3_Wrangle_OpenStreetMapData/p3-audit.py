#auditing street names
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road","Cross","Village","Fold",
            "Trail", "Parkway", "Commons" ,"West" , "East" , "Way" , 'South','Estate' , "Croft" , "Gate","Walk","North","Terrace",
            'Walk' ,'Hill','Grove', 'Mews','Gardens','Park','Centre','Brow','Chase','Arcade','Close','Crescent' ,'Green',
           "Parade" ,"Quay","Grange","House","Knowle","View" ,"Mall" ,"Quays" ,"Meadow" ,"End"]

mapping = { "St": "Street",
            "St.": "Street",
            "street": "Street",
            "Ave": "Avenue",
            "avenue" : "Avenue",
            "Rd." : "Road",
            "road": "Road",
            "Rd": "Road",
            "Gates":"Gate",
            "square" :"Square",
            "east":"East",
            "Meadows" : "Meadow",
            "Cottages":"Cottage",
            "Ends":"End",
            "Orchards": "Orchard",
            "Heyes": "Hey"
            }

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])

    return street_types

#updating street names
def update_name(name, mapping):
     sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
     for abbrv in sorted_keys:
         if(abbrv in name):
             name.rstrip() #removing spaces at the end of the street type
             return name.replace(abbrv, mapping[abbrv])
     return name


def Run_street():
    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name,"=>", better_name
#===========================================================================

def is_phone(elem):
    return (elem.attrib['k'] == "phone")

def is_city(elem): # needed if i will chek manchester
    return (elem.attrib['k'] == "addr:city")
def audit(osmfile):
    osm_file = open(osmfile, "r")
    phone_types = []
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_phone(tag):
                    phone_types.append(tag.attrib['v'])

    return phone_types

def update_phone (phonenumber):
    phone_r = phonenumber.strip()
    phone_s = phone_r.replace(" ", "")
    if phone_s[0] == "+":
        phone_s = phone_s[2:]
        phone_s = re.sub('[^a-zA-Z0-9-_*.]', '', phone_s) #replace special character with "" to remove it
    return phone_s
    
def Run_phone():
    for phone in phones:
        ph = update_phone(phone)
        print phone ,"=>",ph
