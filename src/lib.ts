import { createReadStream } from "fs";
import { createInterface } from "readline";

export class NaiveBayes {
  private _trainingData: string[][] = []
  private _testingData: string[][] = []

  private _trainLabels: number[] = []
  private _testLabels: number[] = []
  private _dictionary = new Map()

  private _featurizedTrain: number[][] = []
  private _featurizedTest: number[][] = []
  private _featurizedPositives: number[][] = []
  private _featurizedNegatives: number[][] = []

  private _numExamples: number = 0
  private _numPos: number = 0
  private _numNeg: number = 0
  private _posPos: number[] = []
  private _posNeg: number[] = []
  private _negPos: number[] = []
  private _negNeg: number[] = []

  constructor(trainFilename: string, testFilename: string, cb: () => void) {
    Promise.all([
      this.loadTrainingData(trainFilename),
      this.loadTestingData(testFilename),
    ]).then((_) => {
      this.makeLabels(this._trainingData, this._trainLabels);
      this.makeLabels(this._testingData, this._testLabels);

      this.makeDictionary();

      this.featurizeAll(this._trainingData, this._trainLabels, this._featurizedTrain)
      this.featurizeAll(this._testingData, this._testLabels, this._featurizedTest)

      this.learnBayes(this._featurizedTrain)

      console.log("The accuracy for the training set is:",
        this.calculateAccuracy(this._featurizedTrain).toFixed(2) + '%');
      console.log("The accuracy for the testing set is:",
        this.calculateAccuracy(this._featurizedTest).toFixed(2) + '%');
    
      cb();
    });
  }

  hypothesizeInput(inputStr: string){
    const parsedInput = this.parseLine(inputStr) 
    const featurizedInput = this.makeFeatureList(parsedInput, -1)
    const hypothesis = this.classify(featurizedInput)
    return hypothesis
  }

  calculateAccuracy(featurizedData: number[][]){
      let correct = 0;
      featurizedData.forEach((example) => {
          let hypothesis = this.classify(example)
          if ( hypothesis === example[example.length - 1]){
              correct += 1;
          }
      })
      return (correct / featurizedData.length * 100)
  }

  classify(featurizedExample: number[]){
      let sumClass0: number = Math.log(this._numNeg / this._numExamples)
      let sumClass1: number = Math.log(this._numPos / this._numExamples)

      try{
        featurizedExample.slice(0,-1).forEach((_, idx) => {
            switch(featurizedExample[idx]){
                case 0: 
                    sumClass0 += this._negNeg[idx];
                    sumClass1 += this._negPos[idx];
                    break;
                case 1:
                    sumClass0 += this._posNeg[idx];
                    sumClass1 += this._posPos[idx];
                    break;
                default:
                    throw "Label error in data (expect 0/1)"
            }
        })
      } catch(err){
          console.error(err);
          console.log("Try cleaning up your featurized data")
          process.exit(1)
      }

      if (sumClass1 > sumClass0) {
          return 1;
      } else {
          return 0;
      }
    }

  tallyOccurances(featurizedData: number[][], occuranceArr: number[]){
      featurizedData.forEach((_, idx) => {
          let jdx = 0;
          this._dictionary.forEach(() => {
              occuranceArr[jdx] += featurizedData[idx][jdx]
              jdx++;
          })
      })
  }

  learnBayes(featurizedData: number[][]){
    featurizedData.forEach(featurizedExample => {
        featurizedExample[featurizedExample.length - 1] === 1 ?
            this._featurizedPositives.push(featurizedExample.slice(0,-1)) :
            this._featurizedNegatives.push(featurizedExample.slice(0,-1)) ;
    })

    this._numPos = this._featurizedPositives.length;
    this._numNeg = this._featurizedNegatives.length;
    this._numExamples = this._numPos + this._numNeg;

    const posOccurance: number[] = [];
    const negOccurance: number[] = [];

    this._dictionary.forEach(() => {
        posOccurance.push(0);
        negOccurance.push(0);
    })

    this.tallyOccurances(this._featurizedPositives, posOccurance);
    this.tallyOccurances(this._featurizedNegatives, negOccurance);

    posOccurance.forEach((_, idx) => {
        this._posPos.push(Math.log((posOccurance[idx] + 1) / (this._numPos + 2)))
        this._negPos.push(Math.log(1 - ((posOccurance[idx] + 1) / (this._numPos + 2)) ))
    })

    negOccurance.forEach((_, idx) => {
        this._posNeg.push(Math.log((negOccurance[idx] + 1) / (this._numNeg + 2)))
        this._negNeg.push(Math.log(1 - ((negOccurance[idx] + 1) / (this._numNeg + 2)) ))
    })
  }

  makeFeatureList(example: string[], exampleLabel: number){
      const features: number[] = []
      this._dictionary.forEach((_, key: string) => {
          if (example.includes(key)){
            features.push(1);
          } else {
            features.push(0);
          }
      } )
      features.push(exampleLabel)
      return features;
  }

  featurizeAll(exampleSet: string[][], labels: number[], featurizedDataArr: number[][]){
    exampleSet.forEach((example: string[], idx) => {
        featurizedDataArr.push(this.makeFeatureList(example, labels[idx]))
    })
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

  parseLine(line: string) {
    const parsed = line
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/[ \t]+/);
    parsed.pop();
    return parsed;
  }

  makeLabels(exampleArr: string[][], labelArr: number[]) {
    exampleArr.forEach((example: string[]) => {
      labelArr.push(parseInt(example.pop()));
    });
  }

  loadFileIntoArr(filename: string, arr: Array<string[]>) {
    return new Promise((resolve) => {
      createInterface({
        input: createReadStream(__dirname + "/../" + filename),
        terminal: false,
      })
        .on("line", (line) => {
          arr.push(this.parseLine(line));
        })
        .on("close", () => {
          resolve();
        });
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