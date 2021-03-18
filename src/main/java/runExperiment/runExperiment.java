package runExperiment;

import experiments.Experiments.ExperimentalArguments;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import java.util.ArrayList;

public class runExperiment{
	public static void main(String[] args) throws Exception {
		 String[] mtsProblems = {
				 "Heartbeat",
				 "SelfRegulationSCP1",
				 "FaceDetection",
				 "SelfRegulationSCP2",
				 "MotorImagery",
				 "FingerMovements"
		 };
		
		String[] classifiers = {
				"DTW_A",
				"CIF",
				"Rocket",
		};

		for (String problem : mtsProblems) {
			for (String classifier : classifiers) {			    
			    System.out.println(problem + "\t" + classifier);
				
				ExperimentalArguments exp = new ExperimentalArguments ();
				exp.dataReadLocation = "src/main/java/experiments/data/mtsc/";
				exp.resultsWriteLocation = "results/";
				exp.classifierName = classifier;
				exp.datasetName = problem;
				exp.foldId = 1;
				exp.generateErrorEstimateOnTrainSet = false;
				exp.run();
				Thread thread = new Thread(exp);
				thread.start();
			}
		}
		//TSCHIEFWrapper.main(null);
		
	}
}
