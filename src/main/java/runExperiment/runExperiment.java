package runExperiment;

import experiments.Experiments.ExperimentalArguments;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import java.util.Properties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class runExperiment{
	public static void main(String[] args) throws Exception {

        int numFolds = 10;
        
        Boolean uts_run = false;
        String uts_dataSource = "ARIMA";
        String uts_resultsWriteLocation = "results/";
        String uts_dataReadLocation;
        if (uts_dataSource == "ARIMA") {
        	uts_dataReadLocation = "src/main/java/experiments/data/tsc/generated/";
        } else {
        	uts_dataReadLocation = "src/main/java/experiments/data/tsc/Univariate_arff/";
        }
        String[] uts_problems = {
        		"Earthquakes",
        		"ECG200",
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
        
        Boolean mts_run = false;
        
        String[] mts_problems = {
				 "Heartbeat",
				 "SelfRegulationSCP1",
				 "SelfRegulationSCP2",
				 "FingerMovements",
				 "MotorImagery",
				 "FaceDetection"
		 };
		
		String[] mts_classifiers = {
				"CBOSS",
				"DTW_D",
				"DTW_A",
				"CIF",
				"ROCKET",
				"MUSE"
		};
        
		String mts_dataReadLocation = "src/main/java/experiments/data/mtsc/";
		String mts_resultsWriteLocation = "results/";
        
        if (args != null) {
    		System.out.println("reading config file");
        	Properties prop = new Properties();
        	String fileName = args[0];
        	try (FileInputStream fis = new FileInputStream(fileName)) {
        	    prop.load(fis);
        	    
        	    uts_run = Boolean.parseBoolean(prop.getProperty("uts.run"));
        	    uts_dataSource = prop.getProperty("uts.dataSource");
        	    uts_dataReadLocation = prop.getProperty("uts.dataReadLocation");
        	    uts_resultsWriteLocation = prop.getProperty("uts.resultsWriteLocation");
        	    uts_problems = prop.getProperty("uts.problems").split(",");
        	    
        	    mts_run = Boolean.parseBoolean(prop.getProperty("mts.run"));
        	    mts_dataReadLocation = prop.getProperty("mts.dataReadLocation");
        	    mts_resultsWriteLocation = prop.getProperty("mts.resultsWriteLocation");
        	    mts_classifiers = prop.getProperty("mts.classifiers").split(",");
        	    mts_problems = prop.getProperty("mts.problems").split(",");
        	    
        	    numFolds = Integer. parseInt(prop.getProperty("numFolds"));
        	    
        		System.out.println(uts_run);
        		System.out.println("uts data source: " + uts_dataSource);
        		System.out.println("uts data path: " + uts_dataReadLocation);
        		System.out.println("uts result path: " + uts_resultsWriteLocation);
        		System.out.println();
        		System.out.println(mts_run);
        		System.out.println("mts data path: " + mts_dataReadLocation);
        		System.out.println("mts result path: " + mts_resultsWriteLocation);
        		System.out.println();
        	    
        	} catch (FileNotFoundException ex) {
        		System.out.println("file not found error parsing run_exp.config");
        		System.out.println(ex);
        	} catch (IOException ex) {
        		System.out.println("io exception parsing run_exp.config");
        		System.out.println(ex);
        	}
        }
        
        
        // running the univariate TSC TS-CHIEF
		if (uts_run) {
	        String[] TSCHIEFWrapper_args = new String[] {
					uts_dataSource,
					uts_resultsWriteLocation,
					uts_dataReadLocation,
					String.join(",", uts_problems),
					String.valueOf(numFolds)
			};
			
			TSCHIEFWrapper.main(TSCHIEFWrapper_args);
		}

		// running multivariate TSC
		if (mts_run) {
			for (String classifier : mts_classifiers ) {
				for (String problem : mts_problems) {			    
				    System.out.println("problem: " + problem + "\t classifier: " + classifier);
					ExperimentalArguments exp = new ExperimentalArguments ();
					exp.dataReadLocation = mts_dataReadLocation;
					exp.resultsWriteLocation = mts_resultsWriteLocation;
					exp.classifierName = classifier;
					exp.datasetName = problem;
					exp.foldId = numFolds;
					exp.generateErrorEstimateOnTrainSet = false;
					Thread thread = new Thread(exp);
					thread.start();
				}
			}
		}		
	}
}
