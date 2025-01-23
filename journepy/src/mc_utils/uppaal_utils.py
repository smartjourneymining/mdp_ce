import networkx as nx

def to_uppaal(g, name, layout = "sfdp", debug = False):
    """Construction of uppaal model (write model into upaal file).
    Assumed fields:
    - cost
    - controllable


    Args:
        g (networkx.DiGraph): Graph to print to uppaal.
        name (Str): file name and path
        layout (str, optional): Layout to layout graph with, "dot" nice to see but expensive to compute. Defaults to "sfdp".
        debug (bool, optional): If set True, prints more additional information. Defaults to False.
    """
    f = open(name, "w+")
    
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog=layout, args='-Grankdir=LR')

    f.write('<?xml version="1.0" encoding="utf-8"?>')
    f.write("<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_1.dtd'>")
    f.write('<nta>')
    f.write('<declaration>')
    f.write('int e = 0;')
    f.write('\n'+'clock x;')
    f.write('\n'+'hybrid clock t;')
    f.write('\n'+'int steps;')
    f.write('\n'+'bool reached_positive = false;')
    f.write('\n'+'bool reached_negative = false;')
    f.write('\n'+'int final_gas = -1;')
    f.write('</declaration>')
    f.write('<template>')
    f.write('<name x="5" y="5">Template</name>')
    
    # print locations
    ids = {}
    for s,i in zip(pos, range(len(pos))):
        ids[s] = i
        print_location(f, "id"+str(i),pos[s][0],pos[s][1],s)
        f.write('\n')
                    
    f.write('<init ref="id'+str(ids['start'])+'"/>')
    
    for e in g.edges:
        assert("cost" in g[e[0]][e[1]] and "controllable" in g[e[0]][e[1]])
        print_edge(f, ids[e[0]], ids[e[1]], pos[e[0]], pos[e[1]], g[e[0]][e[1]]['cost'], g[e[0]][e[1]]['controllable'], e, g)

    f.write('</template>')
    f.write('<system>')
    f.write('Journey = Template();')
    f.write('system Journey;')
    f.write('</system>')
    f.write('</nta>')
    f.close()
    if debug:
        print("all written to", f.name)

def print_location(f, location_id, x, y, name):
    """Helper file to print states

    Args:
        f (file): File writer object to print to
        location_id (int): Location id of current state
        x (int): x_pos
        y (y): y_pos
        name (Str): State name, used to check if final state (contains positive or negative)
    """
    name = str(name)
    name = name.replace('"', '-')
    name = name.replace('{', '')
    name = name.replace('}', '')
    name = name.replace("'", '-')
    name = name.replace("_", '')
    name = name.replace("(", '')
    name = name.replace(")", '')
    f.write('<location id="'+location_id+'" x="'+str(int(x))+'" y="'+str(int(y))+'">')
    f.write('<name x="'+str(int(x))+'" y="'+str(int(y)+20)+'">'+str(name).replace(":", "").replace(" ","").replace(".", "").replace(",", "").replace("-","")+'</name>')
    f.write('<label kind="invariant" x="'+str(int(x))+'" y="'+str(int(y)-30)+'">')
    if "positive" not in name and "negative" not in name and "outOfGas" not in name:
        f.write('x &lt;= ' + str(2))
    else:
        f.write("t'==0")
    f.write('</label>')
    f.write('</location>')

def print_edge(f, s, t, pos_s, pos_t, w, controllable, e, g, guard = False):
    """_summary_

    Args:
        f (file): File to write to
        s (int): start id
        t (int): target id
        pos_s (int): start position
        pos_t (int): target position
        w (int): cost
        controllable (bool): controllable field
        e (_type_): edge
        g (Networks.DiGraph): graph
        guard (bool, optional): _description_. Defaults to False.
    """
    x = (pos_s[0]+pos_t[0])/2
    y = (pos_s[1]+pos_t[1])/2
    if controllable:
        f.write('<transition action = "">')
    else:
        f.write('<transition controllable="false" action = "">')
    f.write('<source ref="id'+str(s)+'"/>')
    f.write('<target ref="id'+str(t)+'"/>')
        
    f.write('<label kind="assignment" x="'+str(int(x))+'" y="'+str(int(y))+'">')
    f.write(' steps += 1')
    f.write(',\n'+ 'x = 0')
    if "positive" in str(e[1]):
        f.write(',\n'+ 'reached_positive = true')
        f.write(',\n'+ 'final_gas = e +'+str(int(round(w))))
    elif "negative" in str(e[1]):
        f.write(',\n'+ 'reached_negative = true')
        f.write(',\n'+ 'final_gas = e + '+str(int(round(w))))
        
    f.write(',\n'+'e = e + '+str(int(round(w))))
    f.write('</label>')
    
    f.write('</transition>')

