const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

function getAllFiles(dirPath, arrayOfFiles) {
  const files = fs.readdirSync(dirPath);
  arrayOfFiles = arrayOfFiles || [];

  files.forEach(function(file) {
    const fullPath = path.join(dirPath, file);
    if (fs.statSync(fullPath).isDirectory()) {
      arrayOfFiles = getAllFiles(fullPath, arrayOfFiles);
    } else {
      arrayOfFiles.push(fullPath);
    }
  });

  return arrayOfFiles;
}

const outDir = path.join(__dirname, 'out');
if (fs.existsSync(outDir)) {
  console.log('Scanning directories in:', outDir);
  const allFiles = getAllFiles(outDir);
  const jsFiles = allFiles.filter(f => f.endsWith('.js'));
  
  console.log(`Found ${jsFiles.length} JavaScript files to obfuscate.`);
  
  jsFiles.forEach(filePath => {
    console.log('Obfuscating:', path.basename(filePath));
    const sourceCode = fs.readFileSync(filePath, 'utf8');
    try {
      const obfuscatedResult = JavaScriptObfuscator.obfuscate(sourceCode, {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 0.5,
        deadCodeInjection: false,
        debugProtection: false,
        disableConsoleOutput: true, // Bloquea impresiones por consola en prod
        identifierNamesGenerator: 'hexadecimal',
        numbersToExpressions: true,
        renameGlobals: false,
        selfDefending: true, // Evita formateadores / prettify externos
        simplify: true,
        splitStrings: true,
        stringArray: true,
        stringArrayEncoding: ['base64'],
        stringArrayThreshold: 0.8,
        unicodeEscapeSequence: false
      });
      fs.writeFileSync(filePath, obfuscatedResult.getObfuscatedCode(), 'utf8');
    } catch (err) {
      console.error(`Error obfuscating ${filePath}:`, err);
    }
  });
  console.log('Obfuscation completed successfully.');
} else {
  console.error('Directory "out/" does not exist. Run "npm run build" first.');
}
