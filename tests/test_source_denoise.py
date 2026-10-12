"""Signal, manifold, provenance and contact regression tests for opt-in filtering."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from omniretargeting.data_sources.source_denoise import (
    debounce_contacts, denoise_parameters, repair_steps, smooth_rotations,
)


def signal():
    fps = 60.
    t = np.arange(600)/fps
    x = np.column_stack([0.12*np.sin(2*np.pi*3*t), 0*t, 1+0*t])
    return fps, t, x


def test_off_identity_and_no_input_mutation():
    fps, t, x = signal()
    r = np.zeros((len(t), 3)); b = np.zeros((len(t), 63))
    before = x.copy()
    a, c, d, report = denoise_parameters(x, r, b, fps)
    assert a is x and c is r and d is b and report is None
    denoise_parameters(x, r, b, fps, 'auto')
    np.testing.assert_array_equal(x, before)


def test_noise_reduction_retains_fast_coherent_motion_and_phase():
    fps, t, clean = signal()
    noisy = clean + np.random.default_rng(42).normal(0, .006, clean.shape)
    filtered, _, _, report = denoise_parameters(noisy, np.zeros((len(t),3)),
                                              np.zeros((len(t),63)),fps,'auto')
    residual = lambda v: np.sqrt(np.mean((v-savgol_filter(v,13,2,axis=0))**2))
    assert residual(filtered) < residual(noisy)*.5
    assert np.max(np.abs(filtered[20:-20,0]-clean[20:-20,0])) < .022
    # Peak timing and amplitude for a true 3 Hz motion, measured independently.
    a = np.column_stack([np.sin(2*np.pi*3*t), np.cos(2*np.pi*3*t)])
    amplitude, phase = np.linalg.lstsq(a, filtered[:,0], rcond=None)[0]
    assert amplitude > .105 and abs(phase) < .002
    import json
    assert json.loads(json.dumps(report))['applied']


def test_clean_fast_motion_is_exactly_unchanged():
    fps, t, clean = signal()
    r = np.column_stack([0*t, 0*t, .5*np.sin(2*np.pi*3*t)])
    b = np.zeros((len(t),63))
    filtered, root, body, report = denoise_parameters(clean,r,b,fps,'auto')
    np.testing.assert_array_equal(filtered,clean)
    np.testing.assert_array_equal(root,r)
    np.testing.assert_array_equal(body,b)
    assert not report['applied']


def test_step_is_spread_and_endpoints_preserved():
    x=np.zeros((240,3));x[120:,2]=-.66
    filtered, jumps=repair_steps(x,60)
    assert len(jumps)==1 and jumps[0]['frame']==120
    assert np.max(np.abs(np.diff(filtered[:,2]))) < .10
    np.testing.assert_array_equal(filtered[[0,-1]],x[[0,-1]])
    assert np.count_nonzero(np.diff(filtered[:,2])) > 20


def test_so3_wrap_is_continuous():
    t=np.arange(240)/60
    angle=2.9+.4*t+np.random.default_rng(5).normal(0,.02,len(t))
    raw=Rotation.from_euler('z',angle[:,None]).as_rotvec()
    out,params=smooth_rotations(raw,60,np.deg2rad(.2))
    rot=Rotation.from_rotvec(out)
    assert np.max((rot[:-1].inv()*rot[1:]).magnitude()) < .04
    assert params[0]['applied']
    assert np.max((Rotation.from_euler('z',(2.9+.4*t)[:,None]).inv()*rot).magnitude()) < .025


def test_debounce_preserves_impact_and_respects_geometry():
    mask=np.zeros(60,bool);mask[5:25]=True;mask[16:19]=False;mask[40:43]=True
    merged,events=debounce_contacts(mask,60)
    assert merged[5:25].all() and not merged[40:43].any()
    assert {e['action'] for e in events}=={'merge_gap','remove_blip'}
    velocity=np.zeros(60);velocity[36:40]=-.3;velocity[43:47]=.3
    merged,events=debounce_contacts(mask,60,velocity)
    assert merged[40:43].all()
    eligible=np.ones(60,bool);eligible[17]=False
    merged,_=debounce_contacts(mask,60,eligible=eligible)
    assert not merged[17]


def test_angular_noise_selects_joints_even_with_clean_translation():
    fps,t,clean=signal()
    body=np.zeros((len(t),63))
    body[:,12:15]=np.random.default_rng(7).normal(0,.02,(len(t),3))
    filtered,root,out,report=denoise_parameters(clean,np.zeros((len(t),3)),body,fps,'auto')
    np.testing.assert_array_equal(filtered,clean)
    np.testing.assert_array_equal(out[:,:12],body[:,:12])
    assert report['applied'] and report['channels']['body_rotations'][4]['applied']
    assert np.std(out[:,12:15]) < np.std(body[:,12:15])*.6


def test_upper_body_only_noise_does_not_request_root_lock():
    fps,t,clean=signal()
    body=np.zeros((len(t),63))
    body[:,57:60]=np.random.default_rng(10).normal(0,.025,(len(t),3))
    _,_,_,report=denoise_parameters(clean,np.zeros((len(t),3)),body,fps,'auto')
    assert report['applied'] and not report['support_requested']


def test_clean_five_hz_motion_survives_despite_high_acceleration():
    fps,t,x=signal()
    x[:,0]=.12*np.sin(2*np.pi*5*t)
    root=np.column_stack([0*t,0*t,.5*np.sin(2*np.pi*5*t)])
    filtered,r,_,report=denoise_parameters(x,root,np.zeros((len(t),63)),fps,'auto')
    np.testing.assert_array_equal(filtered,x)
    np.testing.assert_array_equal(r,root)
    assert not report['applied']


def test_backflip_like_flight_preserves_clean_dynamics_and_removes_added_noise():
    fps = 60.
    t = np.arange(240)/fps
    duration = 0.75
    phase = np.clip((t-1.0)/duration, 0, 1)
    # Ballistic flight, sharp takeoff/landing and a full turn through the pi
    # representation boundary. Large acceleration is intentional, not noise.
    x = np.column_stack([phase, 0*t, 1+4.905*duration**2*phase*(1-phase)])
    turn = 2*np.pi*phase**3*(10+phase*(-15+6*phase))
    clean_rot = Rotation.from_rotvec(np.column_stack([0*t, turn, 0*t]))
    root = clean_rot.as_rotvec()
    body = np.zeros((len(t),63))
    a,r,b,report = denoise_parameters(x,root,body,fps,'auto')
    np.testing.assert_array_equal(a,x)
    np.testing.assert_array_equal(r,root)
    np.testing.assert_array_equal(b,body)
    assert not report['applied']
    assert np.linalg.norm(np.diff(x,n=2,axis=0),axis=1).max()*fps**2 > 100

    rng = np.random.default_rng(31)
    noisy_x = x+rng.normal(0,.006,x.shape)
    noisy_rot = clean_rot*Rotation.from_rotvec(rng.normal(0,.015,root.shape))
    a,r,_,report = denoise_parameters(noisy_x,noisy_rot.as_rotvec(),body,fps,'auto')
    assert report['applied']
    # The ballistic apex and complete turn survive; noise orthogonal to the
    # flight plane and angular reconstruction errors shrink substantially.
    assert np.std(a[:,1]) < np.std(noisy_x[:,1])*.6
    assert abs(a[:,2].max()-x[:,2].max()) < .015
    errors = (clean_rot.inv()*Rotation.from_rotvec(r)).magnitude()
    raw_errors = (clean_rot.inv()*noisy_rot).magnitude()
    assert np.sqrt(np.mean(errors**2)) < np.sqrt(np.mean(raw_errors**2))*.7


def test_contact_table_debounces_only_short_gaps_and_retains_raw_observations(monkeypatch):
    from omniretargeting import source_contacts
    from omniretargeting.surface_contact import surface_contact_config
    frames=40
    raw=np.zeros((frames,1),bool);raw[:30]=True;raw[10:12]=False
    velocity=np.zeros((frames,1,3));velocity[10:12,0,2]=.4;velocity[30:,0,2]=.4
    def observe(*args,**kwargs):
        return dict(distance=np.full((frames,1),.01), normals=np.tile([0,0,1.],(frames,1,1)),
            relative_velocity=velocity.copy(),normal_opposition=np.ones((frames,1)),
            normal_speed=np.abs(velocity[...,2]),active=raw.copy(),
            modes=np.where(raw,'sticking','none'),confidence=raw.astype(float),
            surface_ids=np.zeros((frames,1),int),diagnostics={})
    monkeypatch.setattr(source_contacts,'observe_contacts',observe)
    report=dict(applied=True,contact_merges=[])
    full=dict(points=np.zeros((frames,1,3)),source_denoise=report)
    table=source_contacts.build_source_contacts(full,None,surface_contact_config(),framerate=60)
    assert table.active[:30].all() and not table.active[30:].any()
    np.testing.assert_array_equal(table.observation['active'],raw)
    assert (table.observation['modes'][10:12]=='none').all()
    assert (table.observation['accepted_modes'][10:12]=='touch').all()
    assert (table.observation['accepted_confidence'][10:12]>0).all()
    assert not table.observation['accepted_modes'].flags.writeable
    full['source_denoise']=dict(applied=False)
    off=source_contacts.build_source_contacts(full,None,surface_contact_config(),framerate=60)
    np.testing.assert_array_equal(off.active,raw)
    assert 'accepted_modes' not in off.observation


def _held(values, repeat=2):
    """Zero-order hold of a lower-rate estimate onto the published timeline."""
    return np.repeat(values, repeat, axis=0)


def test_zero_order_hold_is_resampled_not_treated_as_noise():
    fps = 60.
    t30 = np.arange(300)/30.
    x30 = np.column_stack([.3*np.sin(2*np.pi*.8*t30), .1*np.sin(2*np.pi*2*t30), .9+0*t30])
    r30 = np.column_stack([0*t30, 0*t30, .6*np.sin(2*np.pi*.5*t30)])
    x, r = _held(x30), _held(r30)
    b = _held(np.tile(.2*np.sin(2*np.pi*1.2*t30)[:, None], (1, 63)))
    ids = np.repeat(np.arange(300), 2)
    for sample_ids in (ids, None):
        a, root, body, report = denoise_parameters(x, r, b, fps, 'auto', sample_ids=sample_ids)
        assert report['zero_order_hold']['samples'] == 300
        # Truth at published times: hold k is centred at frame 2k+0.5.
        t = (np.arange(600)-.5)/60.
        truth = np.column_stack([.3*np.sin(2*np.pi*.8*t), .1*np.sin(2*np.pi*2*t), .9+0*t])
        assert np.abs(a-truth)[4:-4].max() < .002
        staircase = np.linalg.norm(np.diff(x, 2, axis=0), axis=1).max()*fps**2
        assert np.linalg.norm(np.diff(a, 2, axis=0), axis=1).max()*fps**2 < .1*staircase
        assert np.abs(Rotation.from_rotvec(root).as_euler('xyz')[4:-4, 2]
                      - .6*np.sin(2*np.pi*.5*t[4:-4])).max() < .003


def test_static_clean_motion_is_not_mistaken_for_a_hold():
    t = np.arange(240)/60.
    phase = np.clip((t-1.)/.75, 0, 1)
    x = np.column_stack([phase, 0*t, 1+0*t])
    r = np.column_stack([0*t, 2*np.pi*phase, 0*t])
    a, root, _, report = denoise_parameters(x, r, np.zeros((240, 63)), 60., 'auto')
    assert 'zero_order_hold' not in report
    np.testing.assert_array_equal(a, x)


def test_outlier_sample_is_rejected_not_spread_as_two_steps():
    fps = 30.
    t = np.arange(300)/fps
    clean = np.column_stack([.3*np.sin(2*np.pi*.6*t), 0*t, .9+.05*np.sin(2*np.pi*1.5*t)])
    rng = np.random.default_rng(2)
    noisy = clean + rng.normal(0, .004, clean.shape)
    noisy[150, 2] -= .6
    a, _, _, report = denoise_parameters(noisy, np.zeros((300, 3)), np.zeros((300, 63)), fps, 'auto',
                                         sample_ids=np.arange(300))
    assert not report['jumps']
    assert abs(a[150, 2]-clean[150, 2]) < .02
    assert np.abs(a-clean)[5:-5].max() < .03


def test_noisy_backflip_inside_slow_motion_keeps_flight_and_removes_broadband_noise():
    fps = 60.
    t = np.arange(720)/fps
    slow = np.column_stack([.3*np.sin(2*np.pi*.4*t), .2*np.sin(2*np.pi*.7*t+1), .9+0*t])
    phase = np.clip((t-6.)/.75, 0, 1)
    x = slow.copy()
    x[:, 2] += 4.905*.75**2*phase*(1-phase)
    noisy = x + np.random.default_rng(3).normal(0, .0044, x.shape)
    a, _, _, report = denoise_parameters(noisy, np.zeros((720, 3)), np.zeros((720, 63)), fps, 'auto')
    assert report['channels']['translation']['applied']
    flight = slice(int(6.1*fps), int(6.65*fps))
    assert abs(np.median(np.diff(a[flight, 2], 2))*fps**2 + 9.81) < 1.5
    assert abs(a[:, 2].max()-x[:, 2].max()) < .01
    excess = np.linalg.norm(np.diff(a-x, 2, axis=0), axis=1)[:int(5.5*fps)]*fps**2
    assert np.sqrt(np.mean(excess**2)) < 2.5
