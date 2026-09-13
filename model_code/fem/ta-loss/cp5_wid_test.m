function cp5_wid_test()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'wid_test.log']);
  try
    m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED ARC model %s\n', datestr(now,31));
    fprintf('current wid=%s\n', char(m.param.get('wid')));
    % CHANGE wid 12mm -> 4mm (revert; wid cancels at coil level)
    m.param.set('wid','4[mm]');
    % single point: Npw=10, rho=5000, T=20K, Ip=660 (same as the wid=12mm run)
    m.param('par6').set('T','20[K]'); m.param('par6').set('Ip','660');
    m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false);
    m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/24,td)');
    fprintf('geom (wid=4mm)...\n'); m.geom('geom1').run; fprintf('GEOM_OK\n');
    fprintf('mesh...\n'); m.mesh('mesh1').run; fprintf('MESH_OK\n');
    fprintf('SOLVING... %s\n', datestr(now,31));
    ts=tic; m.sol('sol1').runAll; fprintf('SOLVED %.0fs\n', toc(ts));
    try; m.result.table.create('tw','Table'); catch; end
    m.result.numerical('int4').set('table','tw'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult;
    D=mphtable(m,'tw').data;
    p=max(D(:,end));
    fprintf('=== wid=4mm RESULT ===\n');
    fprintf('mag_peak(wid=4mm) = %.4f W   vs wid=12mm was 2.29 W   ratio=%.3f (expect ~0.33 if wid is culprit)\n', p, p/2.29);
    writematrix(D,[res 'ARC_magloss_wid4mm_Npw10_T20.csv']);
    fprintf('WID_TEST_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'WID_TEST_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
