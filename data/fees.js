const FeesService = require("../services/quote/fees");
const { withExceptionHandler } = require("../utils/decorators");

exports.getFeeConfigs = withExceptionHandler(async (req, res, next) => {
  return await FeesService.getFeeConfigs(req.query.zynkPartnerId || req.partner?.zynkPartnerId || undefined);
}, "Failed to retrieve fee configs");

exports.upsertFeeConfig = withExceptionHandler(async (req, res, next) => {
  const { feeConfigId, zynkPartnerId, infraFeeConfigs, zynkFeeConfigs, rebateConfigs } = req.body || {};
  return await FeesService.upsertFeeConfig(
    feeConfigId,
    zynkPartnerId || req.partner?.zynkPartnerId || null,
    infraFeeConfigs,
    zynkFeeConfigs,
    rebateConfigs
  );
}, "Failed to upsert fee config");
