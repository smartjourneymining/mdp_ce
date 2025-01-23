import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)
import unittest
import networkx as nx
from src.preprocessing import bpic12
from src import game_construction, game_operations

UPPAAL_STRATEGO = "" # set UPPAAL Strateo path here

class TestBPC12Methods(unittest.TestCase):

    def game_equality(self, g1, g2):
        """Test if graphs are equally under considered elements:
        check on crucial elements : action, cost, controllable and color


        Args:
            g1 (networkx.DiGraph): Transition system as directed graph
            g2 (networkx.DiGraph): Transition system as directed graph
        """
        self.assertSetEqual(set(g1.nodes), set(g2.nodes))
        self.assertSetEqual(set(g1.edges), set(g2.edges))
        for e in g1.edges:
            self.assertTrue(g1.edges[e]['action'] == g2.edges[e]['action'])
            self.assertTrue(g1.edges[e]['cost'] == g2.edges[e]['cost'])
            self.assertTrue(g1.edges[e]['controllable'] == g2.edges[e]['controllable'])
            
        for n in g1.nodes:
            if 'decision_boundary' in g1.nodes[n] or 'decision_boundary' in g2.nodes[n]:
                self.assertTrue('decision_boundary' in g1.nodes[n])
                self.assertTrue('decision_boundary' in g2.nodes[n])
                self.assertTrue('positive_guarantee' in g1.nodes[n])
                self.assertTrue('positive_guarantee' in g2.nodes[n])
                self.assertTrue(g1.nodes[n]['decision_boundary'] == g2.nodes[n]['decision_boundary'])
                self.assertTrue(g1.nodes[n]['positive_guarantee'] == g2.nodes[n]['positive_guarantee'])

    
    def test_game_construction(self):
        log = bpic12.preprocessed_log("data/orig/BPI_Challenge_2012.xes")
        g_before = game_construction.game(log, 5, game_construction.ms, actor_path = "data/activities2012.xml")
        g_stored = nx.read_gexf("test/test_bpic12/res/GAME_input:bpi2012_type:multiset_history:5_actors:activities2012.xml.gexf")
        self.game_equality(g_before, g_stored)
        

    def test_decision_boundary_construction(self):
        g_before = nx.read_gexf("test/test_bpic12/res/GAME_input:bpi2012_type:multiset_history:5_actors:activities2012.xml.gexf")
        g_stored = nx.read_gexf("test/test_bpic12/res/DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_.gexf")
        db_game, db = game_operations.compute_db(g_before, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./")
        self.game_equality(db_game, g_stored)

    def test_reduced_decision_boundary_construction(self):
        g_before = nx.read_gexf("test/test_bpic12/res/DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_.gexf")
        g_stored = nx.read_gexf("test/test_bpic12/res/DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_reduced:True.gexf")
        reduced_db_game = game_operations.db_reduction(g_before)
        self.game_equality(reduced_db_game, g_stored)

    def test_static_decision_boundary_construction(self):
        g_before = nx.read_gexf("test/test_bpic12/res/GAME_input:bpi2012_type:multiset_history:5_actors:activities2012.xml.gexf")
        g_stored = nx.read_gexf("test/test_bpic12/res/static_DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_.gexf")
        db_game, db = game_operations.compute_db(g_before, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./", static=True)
        self.game_equality(db_game, g_stored)

    def test_static_reduced_decision_boundary_construction(self):
        g_before = nx.read_gexf("test/test_bpic12/res/DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_.gexf")
        g_stored = nx.read_gexf("test/test_bpic12/res/static_DECB_input:bpi2012_type:multiset_history:5_actors:activities2012_unrolling_factor:0_reduced:True.gexf")
        reduced_db_game = game_operations.db_reduction(g_before, static=True)
        self.game_equality(reduced_db_game, g_stored)    
        
if __name__ == '__main__':
    unittest.main()
