import { NaiveBayes } from './lib';

let a = new NaiveBayes('assets/trainingSet.txt', 'assets/testSet.txt', () => {
    console.log(a.hypothesizeInput("I like this place"))
});