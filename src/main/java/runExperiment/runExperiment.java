package runExperiment;

import experiments.Experiments.ExperimentalArguments;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import java.util.ArrayList;

public class runExperiment{
	public static void main(String[] args) throws Exception {
		
		
		ArrayList<String> mtsProblems = new ArrayList<String>();
		mtsProblems.add("Heartbeat");
		mtsProblems.add("SelfRegulationSCP1");
		mtsProblems.add("FaceDetection");
		mtsProblems.add("SelfRegulationSCP2");
		mtsProblems.add("MotorImagery");
		
		ArrayList<String> classifiers = new ArrayList<String>();
		classifiers.add("DTW_A");
		classifiers.add("CIF");
		
		for (int i = 0; i < mtsProblems.size(); i++) {
		    String problem = mtsProblems.get(i);
			for (int j = 0; j < classifiers.size(); j++) {
			    String classifier = classifiers.get(j);
			    
			    System.out.println(problem + "\t" + classifier);
				
				ExperimentalArguments exp = new ExperimentalArguments ();
				exp.dataReadLocation = "/home/isabella/Documents/TSC/tsml/src/main/java/experiments/data/mtsc";
				exp.resultsWriteLocation = "/home/isabella/Documents/TSC/tsml/results/";
				exp.classifierName = classifier;
				exp.datasetName = problem;
				exp.foldId = 1;
				exp.generateErrorEstimateOnTrainSet = false;
				exp.run();
			}
		}
		
		TSCHIEFWrapper.main(null);
		
	}
}
