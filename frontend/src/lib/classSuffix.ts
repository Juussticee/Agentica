/**
 * Compose a base CSS class plus an allowlisted modifier suffix.
 *
 * The suffix value comes from API data (status/severity/state fields), so it
 * must never be interpolated into the class attribute unchecked. Unknown,
 * missing, or non-string values yield only the neutral base class, which
 * keeps existing styling for every valid value while making class injection
 * and spoofing with arbitrary values impossible.
 */
export function allowlistedClass(
  base: string,
  prefix: string,
  allowed: readonly string[],
  value: unknown,
): string {
  return typeof value === 'string' && allowed.includes(value)
    ? `${base} ${prefix}${value}`
    : base
}
