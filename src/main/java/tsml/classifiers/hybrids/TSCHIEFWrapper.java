package tsml.classifiers.hybrids;
import java.util.Arrays;

import evaluation.MultipleClassifierEvaluation;
import experiments.Experiments;
import tsml.classifiers.MultiThreadable;
import tschief.core.AppContext;
import tschief.core.MultiThreadedTasks;
import tschief.datasets.TSDataset;
import tschief.datasets.TimeSeries;
import tschief.trees.ProximityForest;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

import com.google.common.collect.Lists;

public class TSCHIEFWrapper extends AbstractClassifier implements MultiThreadable {

    private int numThreads = 4;

    private int num_trees = 500;
    private int ee = 5;
    private int boss = 100;
    private int rise = 100;
    private boolean random_dm_per_node = true;

    public ProximityForest pf;

    private int numClasses;
    private Instances header;

    public TSCHIEFWrapper() {
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        this.numThreads = numThreads;
    }

    public int getNum_trees() {
        return num_trees;
    }

    public void setNum_trees(int num_trees) {
        this.num_trees = num_trees;
    }

    public boolean isRandom_dm_per_node() {
        return random_dm_per_node;
    }

    public void setRandom_dm_per_node(boolean random_dm_per_node) {
        this.random_dm_per_node = random_dm_per_node;
    }

    public void setSeed(int seed) {
        AppContext.rand_seed = seed;
        AppContext.rand = new Random(seed);
    }

    public int getSeed() {
        return (int)AppContext.rand_seed;
    }

    private TSDataset toPFDataset(Instances insts) {
        TSDataset dset = new TSDataset(insts.numInstances());

        for (Instance inst : insts) {
            TimeSeries ts = new TimeSeries(getSeries(inst), (int)inst.classValue());
            dset.add(ts);
        }

        return dset;
    }

    private double[] getSeries(Instance inst) {
        double[] d = new double[inst.numAttributes()-1];
        for (int i = 0; i < d.length; i++)
            d[i] = inst.value(i);
        return d;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        //init
        numClasses = data.numClasses();
        header = new Instances(data,0);

        if (AppContext.rand == null) AppContext.rand = new Random();

        AppContext.num_trees = num_trees;
        AppContext.random_dm_per_node = random_dm_per_node;
        AppContext.use_probabilities_to_choose_num_splitters_per_node = false;
        AppContext.num_threads = numThreads;

        AppContext.verbosity = 0;

        ArrayList<AppContext.SplitterType> splitters = new ArrayList<>();
        if (ee > 0) {
            AppContext.ee_splitters_per_node = ee;
            AppContext.ee_enabled = true;
            splitters.add(AppContext.SplitterType.ElasticDistanceSplitter);
        }

        if (boss > 0) {
            AppContext.boss_splitters_per_node = boss;
            AppContext.boss_enabled = true;
            splitters.add(AppContext.SplitterType.BossSplitter);
        }

        if (rise > 0) {
            AppContext.rif_splitters_per_node = rise;
            AppContext.rif_enabled = true;
            splitters.add(AppContext.SplitterType.RIFSplitter);
        }

        AppContext.num_splitters_per_node = ee + boss + rise;
        AppContext.enabled_splitters = splitters.toArray(new AppContext.SplitterType[splitters.size()]);

        pf = new ProximityForest((int)AppContext.rand_seed, new MultiThreadedTasks());

        //actual work
        TSDataset pfdata = toPFDataset(data);

        if (numThreads > 1){
            pf.train_parallel(pfdata);
        }
        else {
            pf.train(pfdata);
        }

    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
//        header.add(inst);
//        Dataset dset = toPFDataset(header);
//        header.remove(0);
//
//        double[] dist = new double[inst.numClasses()];
//        ProximityForestResult pfres = pf.test(dset);
//

        return pf.predict_proba(new TimeSeries(getSeries(inst), -1), numClasses);
    }

//    @Override
//    public double classifyInstance(Instance inst) throws Exception {
//        double[] probs = distributionForInstance(inst);
//
//        int maxClass = 0;
//        for (int n = 1; n < probs.length; ++n) {
//            if (probs[n] > probs[maxClass]) {
//                maxClass = n;
//            }
//            else if (probs[n] == probs[maxClass]){
//                if (AppContext.rand.nextBoolean()){
//                    maxClass = n;
//                }
//            }
//        }
//
//        return maxClass;
//    }

    public static void main(String[] args) throws Exception {

//        ProximityForestWrapper pf = new ProximityForestWrapper();
//        pf.setSeed(0);
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(pf));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(pf, 0.966958211856171, "2019_09_26"));

        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();

        exp.dataReadLocation = "src/main/java/experiments/data/tsc/Univariate_arff/";
//        exp.dataReadLocation = "src/main/java/experiments/data/tsc/generated/";
        exp.resultsWriteLocation = "results/";

        String[] classifiers = { "TSCHIEF" };
        
//        File folder = new File(exp.dataReadLocation);
//        File[] targetFiles = folder.listFiles();
//        
//        String[] datasets = new String[targetFiles.length];
//        int c = 0;
//        for (int i = 0; i < datasets.length; i++) {
//        	String name = targetFiles[i].getName();
//        	String[] meta = name.split("_");
//        	if ( Arrays.asList(300, 1000).contains(Integer.parseInt(meta[2].substring(1)) ) & (Integer.parseInt(meta[2].substring(1)) + Integer.parseInt(meta[3].substring(1)) < 1500 ) & !meta[0].equals("benchmark")){
//        		datasets[c] = name;
//        		c++;
//        		if (name == "ARIMA_S005_N1000_L2500") {
//        			datasets = new String[targetFiles.length];
//        			c = 0;
//        		}
//        			
//        	}
//        }
//        
//        for (int i = 0; i< datasets.length; i++) {
//        	System.out.println(datasets[i]);
//        }
        
        String[] datasets = {
        		//"Earthquakes",
        		//"ECG200",
        		"BeetleFly",
        		"BirdChicken",
        		"Chinatown",
        		"Coffee",
        		"Computers",
        		"DistalPhalanxOutlineCorrect",
        		"DodgerLoopGame",
        		"DodgerLoopWeekend",
        		"ECGFiveDays",
        		"FordA",
        		"FordB",
        		"FreezerRegularTrain",
        		"FreezerSmallTrain",
        		"GunPoint",
        		"GunPointAgeSpan",
        		"GunPointMaleVersusFemale",
        		"GunPointOldVersusYoung",
        		"Ham",
        		"HandOutlines",
        		"Herring",
        		"HouseTwenty",
        		"ItalyPowerDemand",
        		"Lightning2",
        		"MiddlePhalanxOutlineCorrect",
        		"MoteStrain",
        		"PhalangesOutlinesCorrect",
        		"PowerCons",
        		"ProximalPhalanxOutlineCorrect",
        		"SemgHandGenderCh2",
        		"ShapeletSim",
        		"SonyAIBORobotSurface1",
        		"SonyAIBORobotSurface2",
        		"Strawberry",
        		"ToeSegmentation1",
        		"ToeSegmentation2",
        		"TwoLeadECG",
        		"Wafer",
        		"Wine",
        		"WormsTwoClass",
        		"Yoga",
        };
        int numFolds = 5;


        //Because of the static app context, best not run multithreaded, stick to single threaded
        for (String dataset : datasets) {
            for (int f = 0; f < numFolds; f++) {
                exp.datasetName = dataset;
                exp.foldId = f;
                try {
                	Experiments.setupAndRunExperiment(exp);	
                }catch(Exception  E) {
                	System.out.println(E);
                }
            }
        }


//        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(exp.resultsWriteLocation +"ANA/", "sanityCheck", numFolds);
//        mce.setBuildMatlabDiagrams(false);
//        mce.setTestResultsOnly(true);
//        mce.setDatasets(datasets);
//        mce.readInClassifier(exp.classifierName, exp.resultsWriteLocation);
////        mce.readInClassifier("DTWCV", "Z:/Results_7_2_19/FinalisedRepo/"); //no probs, leaving it
//        mce.readInClassifier("PF", "Z:\\Results Working Area\\Benchmarking\\PF\\tsml\\");
//        mce.runComparison();
    }
}
