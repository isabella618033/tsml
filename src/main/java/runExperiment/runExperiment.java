package runExperiment;

import experiments.Experiments.ExperimentalArguments;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import java.util.ArrayList;

public class runExperiment{
	public static void main(String[] args) throws Exception {
		 String[] mtsProblems = {
				 "Heartbeat",
				 "SelfRegulationSCP1",
				 "SelfRegulationSCP2",
				 "FingerMovements",
				 "MotorImagery",
				 "FaceDetection"
		 };
		
		String[] classifiers = {
				//"CBOSS",
				//"DTW_D",
				"DTW_A",
				"CIF",
				"ROCKET",
				"MUSE"
		};

		for (String classifier : classifiers) {
			for (String problem : mtsProblems) {			    
			    System.out.println(problem + "\t" + classifier);
				
				ExperimentalArguments exp = new ExperimentalArguments ();
				exp.dataReadLocation = "src/main/java/experiments/data/mtsc/";
				exp.resultsWriteLocation = "results/";
				exp.classifierName = classifier;
				exp.datasetName = problem;
				exp.foldId = 10;
				exp.generateErrorEstimateOnTrainSet = false;
				//exp.run();
				System.out.println(java.lang.Thread.activeCount());
				Thread thread = new Thread(exp);
				thread.start();
			}
		}
		//TSCHIEFWrapper.main(null);
		
	}
}
