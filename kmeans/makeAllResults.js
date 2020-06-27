const fs = require('fs');
const {spawnSync} = require('child_process');

Array.prototype.last = function() { return this[this.length-1]; }

const configPath = "include/config.hh";
const maxScale = 5;
const makeTargets = [
  'kmeans_sequential'
  ,'kmeans_parallel'
];

main();

function main() {
  backdown(configPath);
  backup(configPath);

  extractDatas(configPath);

  backdown(configPath);
}

function extractDatas(configPath) {
  const threashold = 5

  addNewConfig(configPath, "#define", "SPARSE_LOG");
  addNewConfig(configPath, "#define", "DEEP_LOG");

  makeNewConfig(configPath, 'THREASHOLD', threashold);
  makeNewConfig(configPath, 'DefaultInputFile', '"../mnist/mnist_encoded/encoded_train_ae.npy";');
  
  for(let i=1; i<=maxScale; ++i) {
    const currDir = './results/'+'_theashold'+threashold+ '_scale'+i;
    if(!fs.existsSync(currDir))
      fs.mkdirSync(currDir);

    makeNewConfig(configPath, 'DATA_SCALE', i);

    makeTargets.forEach(target=>{
      const logFile = currDir+'/'+target+ '_theashold'+threashold+ '_scale'+i;

      makeNewConfig(configPath, 'LogFileName', '"'+logFile+'";');

      spawnSync('make', ['clean'], {stdio: 'inherit'});
      spawnSync('make', [target], {stdio: 'inherit'});
    });
  }
}

function addNewConfig(filePath, optionTarget, valueTarget) {
  const configOrigin = fs.readFileSync(filePath).toString('utf-8');
  const lineTarget = optionTarget + " " + valueTarget + "\n";

  let configTarget = configOrigin.split('\n');

  let guard = [];
  guard.push(configTarget.shift());
  while(guard.last() !== '') {
    guard.push(configTarget.shift());
  }

  configTarget.unshift(lineTarget);
  while(guard.length !== 0) {
    configTarget.unshift(guard.pop());
  }

  configTarget = configTarget.join('\n');
  fs.writeFileSync(filePath, configTarget, {encoding:'utf-8'});
}

function makeNewConfig(filePath, optionTarget, valueTarget) {
  const configOrigin = fs.readFileSync(filePath).toString('utf-8');

  const idxOrigin = configOrigin.search(optionTarget);
  const lineOrigin = configOrigin.substr(idxOrigin).split('\n')[0];
  let valueOrigin = lineOrigin.split(' ').last();
  const lineTarget = lineOrigin.replace(valueOrigin, valueTarget);
  const configTarget = configOrigin.replace(lineOrigin, lineTarget);

  fs.writeFileSync(filePath, configTarget, {encoding:'utf-8'});
}

function backup(filePath) {
  const destPath = getBackupPath(filePath);
  fs.copyFileSync(filePath, destPath);
}

function backdown(filePath) {
  const srcPath = getBackupPath(filePath);
  if(fs.exsistsSync(srcPath))
    fs.copyFileSync(srcPath, filePath);
}

function getBackupPath(filePath) {
  const backupFileName = "__temp__" + filePath.split('/').last();
  let backupPath = filePath.split('/');
  backupPath.pop();
  backupPath.push(backupFileName);
  backupPath = backupPath.join('/');
  return backupPath;
}
