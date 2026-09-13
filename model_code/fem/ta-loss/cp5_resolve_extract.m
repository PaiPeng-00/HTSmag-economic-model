function cp5_resolve_extract()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'cp5_resolve_extract.log']);
  try
    m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED %s\n', datestr(now,31));
    m.param('par6').set('Ip','400'); m.param('par6').set('T','20[K]');
    m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false);
    m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/4,td)');
    fprintf('geom...\n'); m.geom('geom1').run; fprintf('mesh...\n'); m.mesh('mesh1').run;
    fprintf('SOLVING (Ip=400,20K,Npw=10)... %s\n', datestr(now,31));
    ts=tic; m.sol('sol1').runAll; fprintf('SOLVED %.0fs (%s)\n', toc(ts), datestr(now,31));
    % --- 正确提取: 数值结果节点 setResult + mphtable ---
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult;
    Dm=mphtable(m,'tmag').data; magpk=max(Dm(:,2));
    fprintf('=== 磁化损耗 (int4, 正确提取) ===\n');
    fprintf('mag vs time: '); fprintf('%.3f ', Dm(:,2)); fprintf('\nmag_peak = %.4f W/coil\n', magpk);
    % 径向损耗 gev3 (若是数值节点)
    radpk=NaN;
    try
      m.result.table.create('trad','Table'); m.result.numerical('gev3').set('table','trad'); m.result.numerical('gev3').set('data','dset1'); m.result.numerical('gev3').setResult;
      Dr=mphtable(m,'trad').data; radpk=max(Dr(:,2));
      fprintf('radial_peak (gev3) = %.3f W/coil\n', radpk);
    catch e1; fprintf('(gev3 提取跳过: %s)\n', e1.message); end
    writematrix(Dm,[res 'ARC_smoke_Ip400_T20_mag.csv']);
    mphsave(m,[base 'ARC_TA_loss_solved_Ip400.mph']);
    fprintf('对比: Ip=660时 mag~2.29W; Ip=400 mag=%.3fW\n', magpk);
    fprintf('SAVED solved mph. RESOLVE_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'RESOLVE_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
