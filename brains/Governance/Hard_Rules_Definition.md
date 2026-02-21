# Hard Rules Definition

Hard rules are non-negotiable constraints enforced by schema and pipeline logic:
1. Schema validation is mandatory before write.
2. Invalid records block pack export.
3. Deprecated records are excluded from default export.
4. Consumer integrations must version-pin exports.
