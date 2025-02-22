#!/usr/bin/env node
const fs = require("fs/promises")
const path = require("path")
const sharp = require("sharp")

if (process.argv.length < 5) {
  console.error("Usage: tile.js <input_image> <output_directory> <number_of_cols>")
  process.exit(1)
}

let input = process.argv[2]
let output_dir = process.argv[3]
let n = process.argv[4]

async function tile() {
  await fs.mkdir(output_dir, { recursive: true })
  let metadata = await sharp(input).metadata()
  let dx = metadata.width / n
  let dy = metadata.height / n
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      await sharp(input)
        .extract({ left: dx * x, top: dy * y, width: dx, height: dy })
        .toFile(path.join(output_dir, `${x}_${y}.jpg`))
    }
  }
}

tile().catch(e => {
  console.error(e)
  process.exit(1)
})
