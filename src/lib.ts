import { createReadStream } from "fs";
import { createInterface } from "readline";

export class NaiveBayes {
  private _trainingData: string[][] = []; // the cleaned up and word separated training data
  private _testingData: string[][] = []; // the cleaned up and word separated testing data

  private _trainLabels: number[] = []; // array for each example's sentiment 1 (positive) or 0 (negative)
  private _testLabels: number[] = []; // array for each example's sentiment 1 (positive) or 0 (negative)
  private _dictionary = new Map(); // all unique words found in the training data

  private _featurizedTrain: number[][] = []; // all examples from training data (featurized)
  private _featurizedTest: number[][] = []; // all examples from testing data (featurized)
  private _featurizedPositives: number[][] = []; // only the positive examples from training data
  private _featurizedNegatives: number[][] = []; // only the negative examples from training data

  private _numExamples: number = 0; // total number of examples in training data
  private _numPos: number = 0; // number of positive examples in training data
  private _numNeg: number = 0; // number of negative examples in training data
  private _posPos: number[] = []; // tally when dictionary word occurred & example was positive 
  private _posNeg: number[] = []; // tally when dictionary word occurred & example was negative 
  private _negPos: number[] = []; // tally when dictionary word didn't occurr & example was positive 
  private _negNeg: number[] = []; // tally when dictionary word didn't occurr & example was positive 


  constructor(trainFilename: string, testFilename: string, cb: () => void) {
    Promise.all([
      // wait for the files to be loaded into memory
      this.loadTrainingData(trainFilename),
      this.loadTestingData(testFilename),
    ]).then((_) => {
      // and then begin training the classifier
      this.makeLabels(this._trainingData, this._trainLabels);
      this.makeLabels(this._testingData, this._testLabels);

      this.makeDictionary();

      // featurize the training data
      this.featurizeAll(
        this._trainingData,
        this._trainLabels,
        this._featurizedTrain
      );

      // featurize the testing data
      this.featurizeAll(
        this._testingData,
        this._testLabels,
        this._featurizedTest
      );

      this.learnBayes(this._featurizedTrain);

      console.log(
        "The accuracy for the training set is:",
        this.calculateAccuracy(this._featurizedTrain).toFixed(2) + "%"
      );
      console.log(
        "The accuracy for the testing set is:",
        this.calculateAccuracy(this._featurizedTest).toFixed(2) + "%"
      );

      cb();
    });
  }

  // loads a file into memory, streaming it line by line
  // file must exist in the ../assets folder
  loadFileIntoArr(filename: string, arr: Array<string[]>) {
    return new Promise((resolve, reject) => {
      createInterface({
        input: createReadStream(__dirname + "/../" + filename),
        terminal: false,
      })
        .on("line", (line) => {
          arr.push(this.parseLine(line));
        })
        .on("close", () => {
          resolve();
        })
        .on("SIGINT", () => {
          reject();
        });
    });
  }

  // remove all punctuation from a string and turn
  // each character lowercase for uniformity
  // finally, split the string into its individual words
  parseLine(line: string) {
    const parsed = line
      .toLowerCase() // 
      .replace(/[^\w\s]/g, "")
      .split(/[ \t]+/);
    parsed.pop();
    return parsed;
  }

  hypothesizeInput(inputStr: string) {
    const parsedInput = this.parseLine(inputStr);
    const featurizedInput = this.makeFeatureList(parsedInput, -1);
    const hypothesis = this.classify(featurizedInput);
    return hypothesis;
  }

  calculateAccuracy(featurizedData: number[][]) {
    let correct = 0;
    featurizedData.forEach((example) => {
      let hypothesis = this.classify(example);
      if (hypothesis === example[example.length - 1]) {
        correct += 1;
      }
    });
    return (correct / featurizedData.length) * 100;
  }

  classify(featurizedExample: number[]) {
    let sumClass0: number = Math.log(this._numNeg / this._numExamples);
    let sumClass1: number = Math.log(this._numPos / this._numExamples);

    try {
      featurizedExample.slice(0, -1).forEach((_, idx) => {
        switch (featurizedExample[idx]) {
          case 0:
            sumClass0 += this._negNeg[idx];
            sumClass1 += this._negPos[idx];
            break;
          case 1:
            sumClass0 += this._posNeg[idx];
            sumClass1 += this._posPos[idx];
            break;
          default:
            throw "Label error in data (expect 0/1)";
        }
      });
    } catch (err) {
      console.error(err);
      console.log("Try cleaning up your featurized data");
      process.exit(1);
    }

    if (sumClass1 > sumClass0) {
      return 1;
    } else {
      return 0;
    }
  }

  tallyOccurances(featurizedData: number[][], occuranceArr: number[]) {
    featurizedData.forEach((_, idx) => {
      let jdx = 0;
      this._dictionary.forEach(() => {
        occuranceArr[jdx] += featurizedData[idx][jdx];
        jdx++;
      });
    });
  }

  learnBayes(featurizedData: number[][]) {
    featurizedData.forEach((featurizedExample) => {
      featurizedExample[featurizedExample.length - 1] === 1
        ? this._featurizedPositives.push(featurizedExample.slice(0, -1))
        : this._featurizedNegatives.push(featurizedExample.slice(0, -1));
    });

    this._numPos = this._featurizedPositives.length;
    this._numNeg = this._featurizedNegatives.length;
    this._numExamples = this._numPos + this._numNeg;

    const posOccurance: number[] = [];
    const negOccurance: number[] = [];

    this._dictionary.forEach(() => {
      posOccurance.push(0);
      negOccurance.push(0);
    });

    this.tallyOccurances(this._featurizedPositives, posOccurance);
    this.tallyOccurances(this._featurizedNegatives, negOccurance);

    posOccurance.forEach((_, idx) => {
      this._posPos.push(Math.log((posOccurance[idx] + 1) / (this._numPos + 2)));
      this._negPos.push(
        Math.log(1 - (posOccurance[idx] + 1) / (this._numPos + 2))
      );
    });

    negOccurance.forEach((_, idx) => {
      this._posNeg.push(Math.log((negOccurance[idx] + 1) / (this._numNeg + 2)));
      this._negNeg.push(
        Math.log(1 - (negOccurance[idx] + 1) / (this._numNeg + 2))
      );
    });
  }

  makeFeatureList(example: string[], exampleLabel: number) {
    const features: number[] = [];
    this._dictionary.forEach((_, key: string) => {
      if (example.includes(key)) {
        features.push(1);
      } else {
        features.push(0);
      }
    });
    features.push(exampleLabel);
    return features;
  }

  featurizeAll(
    exampleSet: string[][],
    labels: number[],
    featurizedDataArr: number[][]
  ) {
    exampleSet.forEach((example: string[], idx) => {
      featurizedDataArr.push(this.makeFeatureList(example, labels[idx]));
    });
  }

  makeDictionary() {
    this._trainingData.forEach((example) => {
      for (let i = 0; i < example.length; i++) {
        if (!this._dictionary.get(example[i])) {
          this._dictionary.set(example[i], null);
        }
      }
    });
  }

  makeLabels(exampleArr: string[][], labelArr: number[]) {
    exampleArr.forEach((example: string[]) => {
      labelArr.push(parseInt(example.pop()));
    });
  }

  getTestingData() {
    return this._testingData;
  }

  async loadTrainingData(filename: string) {
    await this.loadFileIntoArr(filename, this._trainingData);
  }

  async loadTestingData(filename: string) {
    await this.loadFileIntoArr(filename, this._testingData);
  }
}
