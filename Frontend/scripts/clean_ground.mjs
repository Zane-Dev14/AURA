import fs from 'fs';
import path from 'path';

const scenesDir = './components/scenes';
const files = fs.readdirSync(scenesDir).filter(f => f.endsWith('.tsx'));

const grid1 = /<gridHelper[\s\S]*?\/>/g;
const plane1 = /<mesh position={\[0, -1\.6, 0\]}[\s\S]*?<\/mesh>/g;
const plane2 = /<mesh position={\[0, -4, 0\]}[\s\S]*?<\/mesh>/g;

let count = 0;
for (const file of files) {
  const filePath = path.join(scenesDir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  
  const initialLen = content.length;
  content = content.replace(grid1, '');
  content = content.replace(plane1, '');
  content = content.replace(plane2, '');
  
  if (content.length !== initialLen) {
    fs.writeFileSync(filePath, content);
    console.log(`Cleaned safely from ${file}`);
    count++;
  }
}
console.log(`Total completed: ${count} files patched.`);
