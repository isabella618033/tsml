/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities.multivariate_tools;

import utilities.InstanceTools;
import utilities.class_counts.TreeSetClassCounts;
import weka.core.*;
import tsml.transformers.RowNormalizer;

import java.util.ArrayList;

/**
 *
 * @author raj09hxu
 */
public class MultivariateInstanceTools {
    
    
    //given some univariate datastreams, we want to merge them to be interweaved.
    //so given dataset X, Y, Z.
    //X_0,Y_0,Z_0,X_1,.....,Z_m

    //Needs more testing.
    public static Instances mergeStreams(String dataset, Instances[] inst, String[] dimChars){
        
        String name;
        
        Instances firstInst = inst[0];
        int dimensions = inst.length;
        int length = (firstInst.numAttributes()-1)*dimensions;

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            name = dataset + "_" + dimChars[i%dimensions] + "_" + (i/dimensions);
            atts.add(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        atts.add(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        //same number of xInstances 
        Instances result = new Instances(dataset, atts, firstInst.numInstances());

        int size = result.numAttributes()-1;
        
        for(int i=0; i< firstInst.numInstances(); i++){
            result.add(new DenseInstance(size+1));
            
            for(int j=0; j<size;){
                for(int k=0; k< dimensions; k++){
                    result.instance(i).setValue(j,inst[k].get(i).value(j/dimensions)); j++;
                }
            }
        }
        
        for (int j = 0; j < result.numInstances(); j++) {
            result.instance(j).setValue(size, firstInst.get(j).classValue());
        }
        
        return result;
    }
    
    //this function concatinates an array of instances by adding the attributes together. maintains same size in n.
    //assumes properly orderered for class values
    //all atts in inst1, then all atts in inst2 etc.
    public static Instances concatinateInstances(Instances[] data){
        ArrayList<Attribute> atts = new ArrayList();
        String name;
        
        Instances firstInst = data[0];       
        int length =0;
        for (Instances data1 : data) {
            length += data1.numAttributes() - 1;
        }
                
        int dim = 0;
        int localAtt = 0;
        for (int i = 0; i < length; i++) {
            if(i % (length/(firstInst.numAttributes()-1)) == 0){
                    dim++;
                    localAtt=0;
            }
            
            name = "attribute_dimension_" + dim + "_" + localAtt++ + data[0].attribute(0).name();
            atts.add(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        atts.add(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        //same number of xInstances 
        Instances result = new Instances(firstInst.relationName() + "_concatinated", atts, firstInst.numInstances());
        
        for(int i=0; i< firstInst.numInstances(); i++){
            result.add(new DenseInstance(length+1));
            int k=0;
            //for each instance 
            for(Instances inst  : data){
                double[] values = inst.get(i).toDoubleArray();
                for(int j=0; j<values.length - 1; j++){
                    result.instance(i).setValue(k++, values[j]);
                }
            }
        } 
        
        for (int j = 0; j < result.numInstances(); j++) {
            //we always want to write the true ClassValue here. Irrelevant of binarised or not.
            result.instance(j).setValue(length, firstInst.get(j).classValue());
        }
        
        //se the class index where we put it.
        result.setClassIndex(length);
        
        return result;
    }
    
    
    public static Instances createRelationFrom(Instances header, double[][] data){
        int numAttsInChannel = data[0].length;
        Instances output = new Instances(header, data.length);

        //each dense instance is row/ which is actually a channel.
        for(int i=0; i< data.length; i++){
            output.add(new DenseInstance(numAttsInChannel));
            for(int j=0; j<numAttsInChannel; j++)
                output.instance(i).setValue(j, data[i][j]);
        }
        
        return output;
    }

    public static Instances createRelationFrom(Instances header,  ArrayList<ArrayList<Double>> data){
        Instances output = new Instances(header, data.size());

        //each dense instance is row/ which is actually a channel.
        for(int i=0; i< data.size(); i++){
            int numAttsInChannel = data.get(i).size();
            output.add(new DenseInstance(numAttsInChannel));
            for(int j=0; j<numAttsInChannel; j++)
                output.instance(i).setValue(j, data.get(i).get(j));
        }

        return output;
    }
    
    public static Instances createRelationHeader(int numAttsInChannel, int numChannels){
        //construct relational attribute vector.
        ArrayList<Attribute> relational_atts = new ArrayList(numAttsInChannel);
        for (int i = 0; i < numAttsInChannel; i++) {
            relational_atts.add(new Attribute("att" + i));
        }
        
        return new Instances("", relational_atts, numChannels);
    }
/**
 * Input a list of instances, assumed to be properly aligned, where each Instances
 * contains data relating to a single dimension 
 * @param instances: array of Instances
 * @return Instances: single merged file
 */    
    public static Instances mergeToMultivariateInstances(Instances[] instances){
        
        Instance firstInst = instances[0].firstInstance();
        int numAttsInChannel = instances[0].numAttributes()-1;


        ArrayList<Attribute> attributes = new ArrayList<>();
        
        //construct relational attribute.#
        Instances relationHeader = createRelationHeader(numAttsInChannel, instances.length);
        relationHeader.setRelationName("relationalAtt");
        Attribute relational_att = new Attribute("relationalAtt", relationHeader, numAttsInChannel);        
        attributes.add(relational_att);
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        attributes.add(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        Instances output = new Instances("", attributes, instances[0].numInstances());
        
        for(int i=0; i < instances[0].numInstances(); i++){
            //create each row.
            //only two attribtues, relational and class.
            output.add(new DenseInstance(2));
            
            double[][] data = new double[instances.length][numAttsInChannel];
            for(int j=0; j<instances.length; j++)
                for(int k=0; k<numAttsInChannel; k++)
                    data[j][k] = instances[j].get(i).value(k);
            
            //set relation for the dataset/
            Instances relational = createRelationFrom(relationHeader, data);
            
            int index = output.instance(i).attribute(0).addRelation(relational);
            output.instance(i).setValue(0, index);           
            
            //set class value.
            output.instance(i).setValue(1, instances[0].get(i).classValue());
        }
        
        output.setClassIndex(output.numAttributes()-1);
        //System.out.println(relational);
        return output; 
    }
    
    //function which returns the separate channels of a multivariate problem as Instances[].
    public static Instances[] splitMultivariateInstances(Instances multiInstances){
        int d=numDimensions(multiInstances);
        Instances[] output = new Instances[d];
        int length = channelLength(multiInstances); //all the values + a class value.

        //each channel we want to build an Instances object which contains the data, and the class attribute.
        for(int i=0; i< output.length; i++){
            //construct numeric attributes
            ArrayList<Attribute> atts = new ArrayList<>();
            for (int att = 0; att < length; att++) {
                atts.add(new Attribute("channel_"+i+"_"+att));
            }
            
            //construct the class values atttribute.
            Attribute target = multiInstances.attribute(multiInstances.classIndex());
            ArrayList<String> vals = new ArrayList(target.numValues());
            for (int k = 0; k < target.numValues(); k++) {
                vals.add(target.value(k));
            }
            atts.add(new Attribute(multiInstances.attribute(multiInstances.classIndex()).name(), vals));
            
            output[i] = new Instances(multiInstances.relationName() + "_channel_" + i, atts, multiInstances.numInstances());
            output[i].setClassIndex(length);
            
            //for each Instance in 
            for(int j =0; j< multiInstances.numInstances(); j++){
                
                //add the denseinstance to write too.
                output[i].add(new DenseInstance(length+1));

                Instances inst=multiInstances.get(j).relationalValue(0);
                double [] channel = inst.get(i).toDoubleArray();
                int k=0;
                for(; k<channel.length; k++){
                    output[i].instance(j).setValue(k, channel[k]);
                }
                
                double classVal = multiInstances.get(j).classValue();
                output[i].instance(j).setValue(k, classVal);
            }
        }
        
        return output;
    }


    
    public static Instances[] resampleMultivariateInstances(Instances dataset, long seed, double prop){
        Instances[] data_channels = splitMultivariateInstances(dataset);
        Instances[] resample_train_channels = new Instances[data_channels.length];
        Instances[] resample_test_channels = new Instances[data_channels.length];
        
       for (int i = 0; i < resample_train_channels.length; i++) {
            Instances[] temp = utilities.InstanceTools.resampleInstances(data_channels[i], seed, prop);
            resample_train_channels[i] = temp[0];
            resample_test_channels[i] = temp[1];
        }
        
        Instances[] output = new Instances[2];
        output[0] = mergeToMultivariateInstances(resample_train_channels);
        output[1] = mergeToMultivariateInstances(resample_test_channels);
       
        return output;
    }
    

    public static void main(String[] args){
        String local_path = "D:\\Work\\Data\\Multivariate_arff\\"; //Aarons local path for testing.
        String dataset_name = "EigenWorms";
        Instances train = experiments.data.DatasetLoading.loadData(local_path + dataset_name + java.io.File.separator + dataset_name+"_TRAIN.arff");
        Instances test  = experiments.data.DatasetLoading.loadData(local_path + dataset_name + java.io.File.separator + dataset_name+"_TEST.arff");
        Instances[] resampled = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(train, test, 1);
        //Instances[] resampled_old = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances_old(train, test, 1);


        System.out.println(resampled[1].get(resampled[1].numInstances()-1));
        //System.out.println("------------------------------");
        //System.out.println(resampled_old[1].get(resampled_old[1].numInstances()-1));
    }


    /**
     * 
     * This wraps the instancetools functionality for resampling. It is extremely fast compared with the old method.
     * 
     * @param train
     * @param test
     * @param seed
     * @return
    */
    public static Instances[] resampleMultivariateTrainAndTestInstances(Instances train, Instances test, long seed){
        return InstanceTools.resampleTrainAndTestInstances(train, test, seed);
    }


    /**
     * 
     * This function is miles slower. Do not use.
     * 
     * @param train
     * @param test
     * @param seed
     * @return
     */
    @Deprecated
    public static Instances[] resampleMultivariateTrainAndTestInstances_old(Instances train, Instances test, long seed){
        Instances[] train_channels = splitMultivariateInstances(train);
        Instances[] test_channels = splitMultivariateInstances(test);
        
        Instances[] resample_train_channels = new Instances[train_channels.length];
        Instances[] resample_test_channels = new Instances[test_channels.length];
        
        for (int i = 0; i < resample_train_channels.length; i++) {
            System.out.printf("%d / %d \n", i,  resample_train_channels.length);
            Instances[] temp = utilities.InstanceTools.resampleTrainAndTestInstances(train_channels[i], test_channels[i], seed);
            resample_train_channels[i] = temp[0];
            resample_test_channels[i] = temp[1];
        }
        
        Instances[] output = new Instances[2];
        output[0] = mergeToMultivariateInstances(resample_train_channels);
        output[0].setRelationName(train.relationName());
        output[1] = mergeToMultivariateInstances(resample_test_channels);
        output[1].setRelationName(test.relationName());
        
        return output;
    }
    
    
    public static Instance[] splitMultivariateInstanceWithClassVal(Instance instance){
        Instances[] split = splitMultivariateInstances(instance.dataset());
        
        int index = instance.dataset().indexOf(instance);
        
        Instance[] output = new Instance[numDimensions(instance)];
        for(int i=0; i< output.length; i++){
            output[i] = split[i].get(index);
        }  
        return output;
    }
    
    public static Instance[] splitMultivariateInstance(Instance instance){
        Instance[] output = new Instance[numDimensions(instance)];
        for(int i=0; i< output.length; i++){
            output[i] = instance.relationalValue(0).get(i);
        }    
        return output;
    }
    
    
    //this won't include class value.    
    public static double[][] convertMultiInstanceToArrays(Instance[] data){
        double[][] output = new double[data.length][data[0].numAttributes()];
        for(int i=0; i<output.length; i++){
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[i].value(j);
            }
        }
        return output;
    }
    
    //this won't include class value.
    public static double[][] convertMultiInstanceToTransposedArrays(Instance[] data){ 
        double[][] output = new double[data[0].numAttributes()][data.length];
        for(int i=0; i<output.length; i++){
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[j].value(i);
            }
        }
        
        return output;
    }
    
    public static int indexOfRelational(Instances inst, Instances findRelation){
        int index = -1;
        Attribute relationAtt = inst.get(0).attribute(0);
        for(int i=0; i< inst.numInstances(); i++){
            
            if(relationAtt.relation(i).equals(findRelation)){
                index  = i;
                break;
            }
        }
        return index;
    }
    
    public static int numDimensions(Instance multiInstance){
        return multiInstance.relationalValue(0).numInstances();
    }


    public static int numDimensions(Instances multiInstances){
        //get the first attribute which we know is 
        return numDimensions(multiInstances.firstInstance());
    }

    public static int channelLength(Instances multiInstances){
        return channelLength(multiInstances.firstInstance());
    }
    public static int channelLength(Instance multiInstance){
        return multiInstance.relationalValue(0).numAttributes();
    }

 //Tony Added:
/**
 Converts a standard Instances into a multivariate Instances. Assumes each dimension
 * is simply concatenated, so the first dimension is in positions 0 to length-1,
 * second in length to 2*length-1 etc. 
 * First check is that the number of attributes is divisible by length
 */
  public static Instances convertUnivariateToMultivariate(Instances flat, int length){
      int numAtts=flat.numAttributes()-1;
      if(numAtts%length!=0){
          System.out.println("Error, wrong number of attributes "+numAtts+" for problem of length "+length);
          return null;
      }
      int d=numAtts/length;
      System.out.println("Number of atts ="+numAtts+" num dimensions ="+d);
      ArrayList<Attribute> attributes = new ArrayList<>();
        //construct relational attribute.#
        Instances relationHeader = createRelationHeader(length,d);
//                System.out.println(relationHeader);

        relationHeader.setRelationName("relationalAtt");
        Attribute relational_att = new Attribute("relationalAtt", relationHeader, length);        
        attributes.add(relational_att);
        //clone the class values over. 
        Attribute target = flat.attribute(flat.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        attributes.add(new Attribute(flat.attribute(flat.classIndex()).name(), vals));
        Instances output = new Instances(flat.relationName(), attributes, flat.numInstances());
        for(int i=0; i < flat.numInstances(); i++){
            //create each row.
            //only two attribtues, relational and class.
            output.add(new DenseInstance(2));
            
            double[][] data = new double[d][length];
            for(int j=0; j<d; j++){
                for(int k=0; k<length; k++){
                    data[j][k] = flat.get(i).value(j*length+k);
                }
            }            
            //set relation for the dataset/
            Instances relational = createRelationFrom(relationHeader, data);
            
            int index = output.instance(i).attribute(0).addRelation(relational);
            output.instance(i).setValue(0, index);           
            
            //set class value.
            output.instance(i).setValue(1, flat.get(i).classValue());
        }
        
        output.setClassIndex(output.numAttributes()-1);
        return output;          
    }
//Especially for phil :)
  public static Instances transposeRelationalData(Instances data){
       Instances test=data.instance(0).relationalValue(0);
        System.out.println("Number of cases ="+data.numInstances()+" Number of dimensions ="+test.numInstances()+" number of attributes ="+test.numAttributes());
        int d=test.numAttributes();
        int m=test.numInstances();
        int count=0;
        ArrayList<Attribute> attributes = new ArrayList<>();
        Instances relationHeader=MultivariateInstanceTools.createRelationHeader(m,d);
        
         //construct relational attribute.#
        relationHeader.setRelationName("relationalAtt");
        Attribute relational_att = new Attribute("relationalAtt", relationHeader, m);        
        attributes.add(relational_att);
        //clone the class values over. 
        Attribute target = data.attribute(data.classIndex());
        ArrayList<String> vals = new ArrayList<String>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        attributes.add(new Attribute(data.attribute(data.classIndex()).name(), vals));
        Instances output = new Instances(data.relationName(), attributes, data.numInstances());
        for(int i=0; i < data.numInstances(); i++){
            output.add(new DenseInstance(2));
            double[][] raw=new double[d][m];
            test=data.instance(i).relationalValue(0);
            for(int j=0;j<test.numInstances();j++){
                for(int k=0;k<test.instance(j).numAttributes();k++){
                    raw[k][j]=test.instance(j).value(k);//6 dimensions, need to be put in
                }
            }
            //set relation for the dataset/
            Instances relational = createRelationFrom(relationHeader, raw);
            
            int index = output.instance(i).attribute(0).addRelation(relational);
            output.instance(i).setValue(0, index);           
            
            //set class value.
            output.instance(i).setValue(1, data.get(i).classValue());
        }
        output.setClassIndex(output.numAttributes()-1);
        //System.out.println(relational);
        return output;              
  
  }


  public static Instances normaliseDimensions(Instances data) throws Exception {
      Instances[] channels = splitMultivariateInstances(data);
      RowNormalizer norm = new RowNormalizer();
      for (int i = 0; i < channels.length; i++) {
          channels[i] = norm.transform(channels[i]);
      }
      return mergeToMultivariateInstances(channels);
  }

    //function that get a relational instance and return as a set of instances
    public static Instances splitMultivariateInstanceOnInstances(Instance instance){
        Instances output = new Instances("instance", new FastVector(numDimensions(instance)),0);

        for(int i=0; i< instance.relationalValue(0).numAttributes(); i++){
            output.insertAttributeAt(new Attribute("attr" + i), 0);
        }
        for(int i = 0; i< numDimensions(instance); i++){
            output.add(instance.relationalValue(0).get(i));
        }
        output.insertAttributeAt(new Attribute("class"), instance.relationalValue(0).numAttributes());
        output.setClassIndex(output.numAttributes()-1);

        for(int i = 0; i< numDimensions(instance); i++){
            output.get(i).setClassValue(0);
        }
        return output;
    }





}
