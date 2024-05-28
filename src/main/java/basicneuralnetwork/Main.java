package basicneuralnetwork;

import basicneuralnetwork.utilities.FileReaderAndWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Main {
    private static final int MAX_EPOCHS = 600;
    private static double percentForTesting = 0.3;

    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            NeuralNetwork nn = new NeuralNetwork(4, 1,4,3);

            ArrayList<double[]> csvData = FileReaderAndWriter.readCSVFile("src/data/iris.data");

            ArrayList<double[]> train = new ArrayList<>();
            ArrayList<double[]> test = new ArrayList<>();

            splitData(csvData, train, test, percentForTesting);

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                batchTrain(train, nn);
            }

            testNN(test,nn);
        }
    }

    private static void testNN(ArrayList<double[]> test, NeuralNetwork nn) {
        // ** warning **  test does NOT just contain the feature vector.  It's the entire row from the file.
        // make a feature vector that's all the numbers except the last one.
        // get the correct label, which is the last number in the row
        // 

        int correct = 0;
        int total = 0;

        // loop over each element of the test data
        for (double[] p : test) {
            double[] featureVector = Arrays.copyOfRange(p, 0, p.length - 1); // extract the feature vector from the row
            double[] output = nn.guess(featureVector); // run output = nn.guess(...) to get the networks predictions
            
            // find the index of the largest value in output.  That's the guess!
            int guess = 0;
            for (int i = 0; i < output.length; i++) {
                if (output[i] > output[guess]) {
                    guess = i;
                }
            }

            // check if that matches the correct label.  If so add to "correct" counter
            int correctLabel = (int) p[p.length - 1];
            if (guess == correctLabel) {
                correct++;
            }
            total++;

            // display % correct
        }
System.out.println("% correct: " + (double) correct / total * 100.0);
 
        
    }

    private static void batchTrain(ArrayList<double[]> train, NeuralNetwork nn) {
        // ** warning **  test does NOT just contain the feature vector.  It's the entire row from the file.
        // make a feature vector that's all the numbers except the last one.
        // get the correct label, which is the last number in the row

        // loop over each row in train
        for (double[] p : train) {
            double[] featureVector = Arrays.copyOfRange(p, 0, p.length - 1); // extract the feature vector from the row

            // construct the correct output vector
            double[] correctOutput = new double[3];
            correctOutput[(int) p[p.length - 1]] = 1.0;

            // run nn.train(...) to train the network
            nn.train(featureVector, correctOutput);
        }
    }

    /***
     * This method is passed an EMPTY train and test list.  csvData represents all the data.
     * You will randomly divide the data up and add it to train and test.
     * Use percentForTesting to determine what percent of the overall data should get added to test.
     * @param csvData all the data
     * @param train empty list to be filled with training data
     * @param test empty list to be filled with test data
     * @param percentForTesting overall percent to be added to test list
     */
    private static void splitData(ArrayList<double[]> csvData, ArrayList<double[]> train, ArrayList<double[]> test, double percentForTesting){
        Collections.shuffle(csvData);
        // calculate # of items from csvData that should get added to test
        int numTest = (int) (csvData.size() * percentForTesting);
        // add correct # of rows from csvData to test
        for (int i = 0; i < numTest; i++) {
            test.add(csvData.get(i));
        }
        // add the rest to train
        for (int i = numTest; i < csvData.size(); i++) {
            train.add(csvData.get(i));
        }
    }
}
