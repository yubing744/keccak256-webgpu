export type BytesLike = string | number[] | ArrayBuffer | Uint8Array;

export function arrayify(hexString: string): Uint8Array {
    if (hexString.length % 2 !== 0) {
        throw new Error('Invalid hex string');
    }

    let bytes = new Uint8Array(hexString.length / 2);

    for (let i = 0; i < hexString.length; i += 2) {
        bytes[i / 2] = parseInt(hexString.substr(i, 2), 16);
    }

    return bytes;
}

export function isBytesLike(value: any): value is BytesLike {
    return typeof value === 'string' || 
           (Array.isArray(value) && value.every(v => typeof v === 'number')) ||
           value instanceof ArrayBuffer ||
           value instanceof Uint8Array;
}
