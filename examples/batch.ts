import { keccak256_gpu_batch } from '../src';

const messages = [
  new Uint8Array([0x01, 0x00, 0x00, 0x00]), // int 1
  new Uint8Array([0x02, 0x00, 0x00, 0x00]), // int 2
  new Uint8Array([0x03, 0x00, 0x00, 0x00]), // int 3
  new Uint8Array([0x04, 0x00, 0x00, 0x00]), // int 4
  new Uint8Array([0x05, 0x00, 0x00, 0x00]), // int 5
  new Uint8Array([0x06, 0x00, 0x00, 0x00]), // int 6
  new Uint8Array([0x07, 0x00, 0x00, 0x00]), // int 7
  new Uint8Array([0x08, 0x00, 0x00, 0x00]), // int 8
  new Uint8Array([0x09, 0x00, 0x00, 0x00]), // int 9
];

// each message in messages must have the same size
const hashes = await keccak256_gpu_batch(messages);
for (let i = 0; i < messages.length; i++) {
  console.log(
    'message:',
    messages[i].reduce(
      (a: any, b: any) => a + b.toString(16).padStart(2, '0'),
      ''
    )
  );

  console.log(
    'gpu_keccak256:',
    '0x' +
      hashes
        .subarray(i * 32, i * 32 + 32)
        .reduce(
          (a: any, b: any) => a + b.toString(16).padStart(2, '0'),
          ''
        )
  );

  console.log('');
}
