function cp5_smoke_ip400()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'smoke_ip400.log']);
  try
    m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED %s\n', datestr(now,31));
    fprintf('OLD: Ip=%s T=%s Npw=%s wid=%s\n', char(m.param('par6').get('Ip')), ...
            char(m.param('par6').get('T')), char(m.param('par2').get('Npw')), char(m.param.get('wid')));
    % --- 冒烟测试改动: Ip 660->400 (FEM 锁定值), 20K, Npw=10, rho=5000 ---
    m.param('par6').set('Ip','400'); m.param('par6').set('T','20[K]');
    m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false);
    m.study('std1').feature('param1').active(false);
    % 粗时间步求快 (5 步 vs 全跑24步)，只验证 Ip=400 不发散
    m.study('std1').feature('time').set('tlist','range(0,td/4,td)');
    fprintf('geom...\n'); m.geom('geom1').run;
    fprintf('mesh...\n'); m.mesh('mesh1').run;
    fprintf('SMOKE SOLVING (Ip=400,20K,Npw=10, 粗5步)... %s\n', datestr(now,31));
    ts=tic; m.sol('sol1').runAll; fprintf('SMOKE SOLVED %.0fs (%s)\n', toc(ts), datestr(now,31));
    [t,mag]=safe2(@() mphglobal(m,{'t','int4'},'dataset','dset1','solnum','all'));
    [~,rad]=safe2(@() mphglobal(m,{'t','gev3'},'dataset','dset1','solnum','all'));
    fprintf('=== 冒烟结果 (Ip=400, 20K) ===\n');
    fprintf('mag_peak=%.4f W/coil, radial_peak=%.2f W/coil\n', max(mag), max(rad));
    fprintf('(对比 Ip=660 时 mag~2.29W; Ip=400 应更小, 且不发散)\n');
    writematrix([t(:) mag(:) rad(:)], [res 'ARC_smoke_Ip400_T20.csv']);
    fprintf('SMOKE_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'SMOKE_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
function [a,b]=safe2(f); try; r=f(); a=r{1}; b=r{2}; catch; a=NaN; b=NaN; end; end
