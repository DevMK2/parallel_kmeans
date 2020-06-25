'use strict';

const fs = require('fs');
const { spawnSync } = require('child_process');

const watchExecutes = [
  'kmeans', 
  'kmeans_sorting', 
  'kmeans_parallel',
  'kmeans_parallel_const', 
  'kmeans_parallel_sorting', 
  'kmeans_parallel_mempattern',
];

Array.prototype.pjoin = function() { return this.join('/'); }
Array.prototype.last= function() { return this[this.length-1]; }
String.prototype.name = function() { return this.trim().split('/').last().split('.')[0]; }

main();

function main() {
  let files = readdirRecursive(__dirname, ['.hh', '.cc', '.cu']);
  watchFiels(files);
}

function readdirRecursive(path, greps=[]) {
  if(fs.lstatSync(path).isFile()) {
    return [path];
  }

  let files = [];
  fs.readdirSync(path).forEach(file=>{ 
    files = files.concat(readdirRecursive([path, file].pjoin(), greps));
  });

  return files.filter(file=>greps.filter(grep=>file.includes(grep)).length!==0);
}

function watchFiels(files) {
  let executes = files.filter(file=>watchExecutes.filter(watchName=>file.name().includes(watchName)).length!==0);

  executes.forEach(execute=>{
    fs.watchFile(execute, {interval:500}, (curr, prev)=>{
      spawnSync('make', ['clean'], {stdio: 'inherit'});
      spawnSync('make', [execute.name()], {stdio: 'inherit'});
    });
  });
}
