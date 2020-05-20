import { NaiveBayes } from './lib';

const prompt = require('prompt-sync')({sigint: true})

const naiveBayes = new NaiveBayes('assets/trainingSet.txt', 'assets/testSet.txt', () => {

    console.log('\nEnter in your input and the classifier will predict whether the string is positive or negative.')
    console.log('Enter an empty string to quit')

    while(true){
        let userInput = prompt('\n>> ')
        if (userInput === '') { process.exit(0) }

        console.log('\nYour review is predicted to be',
            naiveBayes.hypothesizeInput(userInput) ? 'positive.' : 'negative.')
    }
});