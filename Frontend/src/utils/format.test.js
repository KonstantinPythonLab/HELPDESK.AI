/**
 * @file format.test.js
 * @description Unit tests for format.js utility functions.
 */

import { describe, it, expect } from 'vitest';
import { formatTicketId } from './format';

// ---------------------------------------------------------------------------
// formatTicketId
// ---------------------------------------------------------------------------

describe('formatTicketId', () => {
    it('returns empty string for null', () => {
        expect(formatTicketId(null)).toBe('');
    });

    it('returns empty string for undefined', () => {
        expect(formatTicketId(undefined)).toBe('');
    });

    it('returns empty string for empty string', () => {
        expect(formatTicketId('')).toBe('');
    });

    it('returns short string as-is (length <= 8)', () => {
        expect(formatTicketId('ABC123')).toBe('ABC123');
    });

    it('returns 8-character string as-is', () => {
        expect(formatTicketId('12345678')).toBe('12345678');
    });

    it('extracts and uppercases first segment of UUID', () => {
        expect(formatTicketId('a1b2c3d4-e5f6-7890-abcd-ef1234567890')).toBe('A1B2C3D4');
    });

    it('handles uppercase UUID input', () => {
        expect(formatTicketId('A1B2C3D4-E5F6-7890-ABCD-EF1234567890')).toBe('A1B2C3D4');
    });

    it('handles UUID without dashes', () => {
        // 32-char string without dashes — length > 8, splits on '-' which gives full string
        const uuid32 = 'a1b2c3d4e5f67890abcdef1234567890';
        expect(formatTicketId(uuid32)).toBe(uuid32.toUpperCase());
    });

    it('handles numeric input', () => {
        // Numbers <= 8 digits returned as-is
        expect(formatTicketId(12345)).toBe(12345);
    });

    it('handles numeric input > 8 digits by converting to string', () => {
        const result = formatTicketId(123456789);
        expect(typeof result).toBe('string');
    });

    it('returns boolean false as empty string (falsy)', () => {
        expect(formatTicketId(false)).toBe('');
    });

    it('returns zero as empty string (falsy)', () => {
        expect(formatTicketId(0)).toBe('');
    });
});
