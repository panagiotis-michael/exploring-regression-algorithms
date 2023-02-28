import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.RBFRegressor;
import weka.core.Instances;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.NumericPrediction;
import javax.swing.JFrame;

public class rbf {
    public static void main(String args[]) throws Exception{
        // Read selwood arff file
        BufferedReader selwood = new BufferedReader(new FileReader("selwood.arff"));
        
        // Create selwood dataset and select class label
        Instances train_data_selwood = new Instances(selwood);
        train_data_selwood.setClassIndex(train_data_selwood.numAttributes()-1);
        
        // Set hyperparameters for RBF network
        RBFRegressor rbf = new RBFRegressor();
        rbf.setBatchSize("1");
        rbf.setNumDecimalPlaces(2);
        rbf.setNumFunctions(6);
        rbf.setNumThreads(1);
        rbf.setPoolSize(1);
        rbf.setRidge(0.1);
        rbf.setTolerance(0.00001);
        rbf.setSeed(1);
        rbf.setUseAttributeWeights(true);
        rbf.setUseCGD(true);
        rbf.setUseNormalizedBasisFunctions(true);
        
        // Get the evaluation from the selwood model using 10 - fold cross validation
        Evaluation eval_selwood = new Evaluation(train_data_selwood);
        eval_selwood.crossValidateModel(rbf, train_data_selwood, 10, new Random(1));
        
        // Print results
        System.out.println(eval_selwood.toSummaryString("\n***** RBF Selwood *****\n", true));
        
        // Read fish arff file
        BufferedReader fish = new BufferedReader(new FileReader("fish.arff"));
        
        // Create fish dataset and select class label
        Instances train_data_fish = new Instances(fish);
        train_data_fish.setClassIndex(train_data_fish.numAttributes()-1);
        
        // Get the evaluation from the fish model using 10 - fold cross validation
        Evaluation eval_fish = new Evaluation(train_data_fish);
        eval_fish.crossValidateModel(rbf, train_data_fish, 10, new Random(1));
        
        // Print results
        System.out.println(eval_fish.toSummaryString("\n***** RBF Fish *****\n", true));
        
        // Get predictions from both models
        ArrayList<Prediction> selwood_predictions = eval_selwood.predictions();
        ArrayList<Prediction> fish_predictions = eval_fish.predictions();
        
        // Create arrays to get the predicted and true values for plotting
        double[] selwood_y_pred = new double[selwood_predictions.size()];
        double[] selwood_y_true = new double[selwood_predictions.size()];
        
        double[] fish_y_pred = new double[fish_predictions.size()];
        double[] fish_y_true = new double[fish_predictions.size()];
        
        // Getting the values for both models
        for (int i = 0; i < selwood_predictions.size(); i++) {
            NumericPrediction prediction = (NumericPrediction) selwood_predictions.get(i);
            selwood_y_pred[i] = prediction.predicted();
            selwood_y_true[i] = prediction.actual();
        }
        
        for (int i = 0; i < fish_predictions.size(); i++) {
            NumericPrediction prediction = (NumericPrediction) fish_predictions.get(i);
            fish_y_pred[i] = prediction.predicted();
            fish_y_true[i] = prediction.actual();
        }
        
        // Setting up the charts (scatter plots)
        XYSeries selwood_series = new XYSeries("Values");
        for (int i = 0; i < selwood_predictions.size(); i++) {
            selwood_series.add(selwood_y_true[i], selwood_y_pred[i]);
        }
        XYDataset selwood_dataset = new XYSeriesCollection(selwood_series);
        JFreeChart selwood_chart = ChartFactory.createScatterPlot("RBF Selwood Scatter Plot", "Actual class values", "Predicted class values", selwood_dataset);
        ChartPanel selwood_panel = new ChartPanel(selwood_chart);
        JFrame selwood_frame = new JFrame(" RBF Selwood Scatter Plot");
        selwood_frame.setContentPane(selwood_panel);
        selwood_frame.pack();
        selwood_frame.setVisible(true);
        
        XYSeries fish_series = new XYSeries("Values");
        for (int i = 0; i < fish_predictions.size(); i++) {
            fish_series.add(fish_y_true[i], fish_y_pred[i]);
        }
        XYDataset fish_dataset = new XYSeriesCollection(fish_series);
        JFreeChart fish_chart = ChartFactory.createScatterPlot("RBF Fish Scatter Plot", "Actual class values", "Predicted class values", fish_dataset);
        ChartPanel fish_panel = new ChartPanel(fish_chart);
        JFrame fish_frame = new JFrame("RBF Fish Scatter Plot");
        fish_frame.setContentPane(fish_panel);
        fish_frame.pack();
        fish_frame.setVisible(true);
    }
}
